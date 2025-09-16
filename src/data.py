from pathlib import Path
import itertools
import joblib
import random

import scipy
import pandas as pd
import numpy as np
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from torch.utils.data import Dataset
import lightning as L
import torch
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from src.augmentations import get_augmentations, AugmentationList


sensor_sizes_dict = {
    "AIRS-CH0": [[11250, 32, 356], [32, 356]],
    "FGS1": [[135000, 32, 32], [32, 32]],
}

SIGMA = 60

def dgauss(sig):
    xs = np.arange(-3.*sig, 3.*sig+1)
    den = 2.*sig*sig
    ys = np.exp(-np.square(xs)/den)
    dys = -2*xs/den*ys
    return dys

def d2gauss(sig):
    xs = np.arange(-3.*sig, 3.*sig+1)
    den = 2.*sig*sig
    ys = np.exp(-np.square(xs)/den)
    d2ys = np.square(2/den)*ys*(xs-sig)*(xs+sig)
    return d2ys

def find_transit_edges(S, sigma):
    """ Find the centers of the transitions """

    Sc = np.convolve(S, dgauss(sigma), mode="valid")
    off = int((S.size-Sc.size)/2)
    mid = Sc.size//2
    
    transit_start = np.argmin(Sc[3:mid-3])+off+3
    transit_end = np.argmax(Sc[mid+3:-3])+off+mid+3

    return transit_start, transit_end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def find_transit_slopes(S, transit_start, transit_end, sigma):
    """find the width of the transitions"""
    
    Sc2 = np.convolve(S, d2gauss(sigma), mode="valid")
    off = int((S.size-Sc2.size)/2)

    t1 = transit_start - off
    t2 = transit_end - off
    
    sz = 2*sigma

    try:
        t1a = np.argmin(Sc2[t1-sz:t1+1])+t1-sz+off
    except Exception:
        t1a = None

    try:
        t1b = np.argmax(Sc2[t1:t1+sz+1])+t1+off
    except Exception:
        t1b = None

    try:
        t2a = np.argmax(Sc2[t2-sz:t2+1])+t2-sz+off
    except Exception:
        t2a = None

    try:
        t2b = np.argmin(Sc2[t2:t2+sz+1])+t2+off
    except Exception:
        t2b = None

    return t1a, t1b, t2a, t2b


# ----------------------------------------

class StaticModelConfig:
    SCALE = 0.93960
    SIGMA = 0.0009
    
    CUT_INF = 39
    CUT_SUP = 250
        
    MODEL_PHASE_DETECTION_SLICE = slice(30, 140)
    MODEL_OPTIMIZATION_DELTA = 7
    MODEL_POLYNOMIAL_DEGREE = 3
    

class StaticModel:
    def __init__(self, config):
        self.cfg = config

    def _phase_detector(self, signal):
        search_slice = self.cfg.MODEL_PHASE_DETECTION_SLICE
        min_index = np.argmin(signal[search_slice]) + search_slice.start
        
        signal1 = signal[:min_index]
        signal2 = signal[min_index:]

        grad1 = np.gradient(signal1)
        grad1 /= grad1.max()
        
        grad2 = np.gradient(signal2)
        grad2 /= grad2.max()

        phase1 = np.argmin(grad1)
        phase2 = np.argmax(grad2) + min_index

        return phase1, phase2
    
    def _objective_function(self, s, signal, phase1, phase2):
        delta = self.cfg.MODEL_OPTIMIZATION_DELTA
        power = self.cfg.MODEL_POLYNOMIAL_DEGREE

        if phase1 - delta <= 0 or phase2 + delta >= len(signal) or phase2 - delta - (phase1 + delta) < 5:
            delta = 2

        y = np.concatenate([
            signal[: phase1 - delta],
            signal[phase1 + delta : phase2 - delta] * (1 + s),
            signal[phase2 + delta :]
        ])
        x = np.arange(len(y))

        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        
        return error

    def predict_spectre(self, single_preprocessed_signal) -> np.ndarray:
        output = []
        for i in range(single_preprocessed_signal.shape[-1]):
            signal_1d = single_preprocessed_signal[:, i]
            signal_1d = savgol_filter(signal_1d, 30, 2)
            
            phase1, phase2 = self._phase_detector(signal_1d)

            phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
            phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)    

            result = minimize(
                fun=self._objective_function,
                x0=[0.0001],
                args=(signal_1d, phase1, phase2),
                method="Nelder-Mead"
            )
            output.append(np.clip(result.x[0], 0.0, None))
        
        return np.array(output[::-1])
    
    def predict_static(self, single_preprocessed_signal) -> float:
        signal_1d = single_preprocessed_signal[:, :-1].mean(axis=1)
        signal_1d = savgol_filter(signal_1d, 30, 2)
        
        phase1, phase2 = self._phase_detector(signal_1d)

        phase1 = max(self.cfg.MODEL_OPTIMIZATION_DELTA, phase1)
        phase2 = min(len(signal_1d) - self.cfg.MODEL_OPTIMIZATION_DELTA - 1, phase2)    

        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],
            args=(signal_1d, phase1, phase2),
            method="Nelder-Mead"
        )
        
        return result.x[0]



class SensorData:
    def __init__(
            self, 
            path: Path | str, 
            planet_id: int | str, 
            sensor: str, 
            transit_num: int | str,
            axis_info: pd.DataFrame | None = None
            ) -> None:
        if sensor not in ["AIRS-CH0", "FGS1"]:
            raise ValueError("Invalid sensor")
        
        if sensor == "AIRS-CH0":
            self.bin_coef = 15
        else:
            self.bin_coef = 180
        
        self.path = Path(path)
        self.sensor = sensor
        self.planet_id = planet_id
        self.transit_num = transit_num
        self.gain = 0.4369
        self.offset = -1000.0
        self.edges = {}
        
        self.signal = self._read_signal()
        self.calibration_folder = f"{sensor}_calibration_{transit_num}"

        self._construct_dt(axis_info)

        self.dark_frame = self._read_calibration("dark.parquet").reshape(sensor_sizes_dict[sensor][1])
        self.dead_frame = self._read_calibration("dead.parquet", convert=False).reshape(sensor_sizes_dict[sensor][1])
        self.flat_frame = self._read_calibration("flat.parquet").reshape(sensor_sizes_dict[sensor][1])
        self.read_frame = self._read_calibration("read.parquet").reshape(sensor_sizes_dict[sensor][1])
        self.linear_corr_frame = self._read_calibration("linear_corr.parquet").reshape([6] + sensor_sizes_dict[sensor][1])

        self.adc_converted = False
        self.mask_hot_dead_converted = False
        self.linear_corr_converted = False
        self.clean_dark_converted = False
        self.cds_converted = False
        self.flat_field_converted = False
        self.read_converted = False
        self.binned = False
        self.interpolated = False

    def _construct_dt(self, axis_info: pd.DataFrame | None = None) -> None:
        if self.sensor == "AIRS-CH0":
            dt = axis_info["AIRS-CH0-integration_time"].dropna().values
            dt[1::2] += 4.5
        else:
            dt = np.ones(len(self.signal)) * 0.1
            dt[1::2] += 0.1

        self.dt = dt

    def _read_calibration(self, filename: str, convert: bool = True) -> np.ndarray:
        a = pd.read_parquet(
            self.path / str(self.planet_id) / self.calibration_folder / filename,
            engine="pyarrow",
        ).values
        
        if convert:
            a = a.astype(np.float64)
        
        return a
    
    def _read_signal(self) -> np.ndarray:
        return pd.read_parquet(
            self.path / str(self.planet_id) / f"{self.sensor}_signal_{self.transit_num}.parquet",
            engine="pyarrow",
        ).values.astype(np.float64).reshape(sensor_sizes_dict[self.sensor][0])

    def apply_adc_convert(self) -> None:
        if self.adc_converted:
            return
        
        self.signal /= self.gain
        self.signal += self.offset
        self.adc_converted = True

    def apply_mask_hot_dead(self) -> None:
        if self.mask_hot_dead_converted:
            return
        
        hot = sigma_clip(self.dark_frame, sigma=5, maxiters=5).mask
        hot = np.tile(hot, (self.signal.shape[0], 1, 1))
        dead = np.tile(self.dead_frame, (self.signal.shape[0], 1, 1))

        # Set values to np.nan where dead or hot pixels are found
        self.signal[dead] = np.nan
        self.signal[hot] = np.nan
        
        self.mask_hot_dead_converted = True

    def apply_linear_corr(self) -> None:
        if self.linear_corr_converted:
            return
        
        linear_corr = np.flip(self.linear_corr_frame, axis=0)
        for x, y in itertools.product(
                    range(self.signal.shape[1]), range(self.signal.shape[2])
                ):
            poli = np.poly1d(linear_corr[:, x, y])
            self.signal[:, x, y] = poli(self.signal[:, x, y])

        self.linear_corr_converted = True

    def apply_clean_dark(self) -> None:
        if self.clean_dark_converted:
            return
        
        dark = np.tile(self.dark_frame, (self.signal.shape[0], 1, 1))
        self.signal -= dark * self.dt[:, np.newaxis, np.newaxis]

        self.clean_dark_converted = True

    def apply_cds(self) -> None:
        if self.cds_converted:
            return
        
        self.signal = np.subtract(self.signal[1::2, :, :], self.signal[::2, :, :])
        self.cds_converted = True

    def apply_correct_flat_field(self) -> None:
        if self.flat_field_converted:
            return
        
        self.signal = self.signal / self.flat_frame
        self.flat_field_converted = True

    def apply_clean_read(self) -> None:
        if self.read_converted:
            return
        
        read = np.tile(self.read_frame * 2, (self.signal.shape[0], 1, 1))
        self.signal -= read
        self.read_converted = True

    def fill_nan(self) -> None:
        self.signal = np.nan_to_num(self.signal, nan=np.nanmean(self.signal))

    def fill_nans_with_interpolation(self) -> None:
        if self.interpolated:
            return
        
        # Get the shape of the signal
        time_steps, _, _ = self.signal.shape
                
        for t in range(time_steps):
            # Get the current frame
            frame = self.signal[t]
            
            # Create a mask for NaNs
            nan_mask = np.isnan(frame)
            
            if np.all(nan_mask):  # Check if all values are NaN
                continue
            
            # Get coordinates of valid points
            x, y = np.indices(frame.shape)
            valid_points = ~nan_mask
            
            # Prepare points for interpolation
            points = np.array([x[valid_points], y[valid_points]]).T
            values = frame[valid_points]
            
            # Interpolate over the grid
            filled_frame = scipy.interpolate.griddata(points, values, (x, y), method='linear')
            
            # Fill NaNs in the filled_signal
            self.signal[t] = np.where(nan_mask, filled_frame, frame)
        
        self.interpolated = True
        

    def process(self) -> None:
        self.apply_adc_convert()
        # self.apply_mask_hot_dead()
        # self.apply_linear_corr()

        # self.apply_clean_dark()
        # self.apply_clean_read()

        self.apply_cds()

        if self.sensor == "AIRS-CH0":
            self.find_edges()
        else:
            self.set_edges(None)
        
        self.bin_obs(self.bin_coef)

        # self.apply_correct_flat_field()
        # self.fill_nans_with_interpolation()
        self.fill_nan()

    def bin_obs(self, binning: int) -> None:
        if self.binned:
            return
        
        cds_transposed = self.signal[None, ...].transpose(0,1,3,2)
        cds_binned = np.zeros((cds_transposed.shape[0], cds_transposed.shape[1]//binning, cds_transposed.shape[2], cds_transposed.shape[3]))
        for i in range(cds_transposed.shape[1]//binning):
            cds_binned[:,i,:,:] = np.sum(cds_transposed[:,i*binning:(i+1)*binning,:,:], axis=1)
        
        self.signal = cds_binned.transpose(0,1,3,2)[0]

        for k, v in self.edges.items():
            self.edges[k] = v // binning if v else None
        self.binned = True

    def plot_raw(self, time: int = 0) -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(self.signal[time], aspect="auto")
        ax.set_title(f"Planet {self.planet_id} {self.sensor} {time = }")
        return fig
    
    def plot_curve(self) -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        light_curve = np.nan_to_num(self.signal).sum(axis=(1,2))
        ax.plot(light_curve/light_curve.mean(), '-')

        # plot edges
        if self.edges:
            ax.axvline(self.edges["transit_start"], color="g", alpha=0.5)
            ax.axvline(self.edges["transit_end"], color="g", alpha=0.5)
            if self.edges["t1a"]:
                ax.axvline(self.edges["t1a"], color='blue')

            if self.edges["t1b"]:
                ax.axvline(self.edges["t1b"], color='blue')

            if self.edges["t2a"]:
                ax.axvline(self.edges["t2a"], color='blue')

            if self.edges["t2b"]:
                ax.axvline(self.edges["t2b"], color='blue')
        
        ax.set_title(f"Light curve for planet {self.planet_id} {self.sensor}")
        return fig
    
    def find_edges(self) -> None:
        light_curve = np.nan_to_num(self.signal).sum(axis=(1,2))
        light_curve = light_curve / light_curve.mean()

        transit_start, transit_end = find_transit_edges(light_curve, sigma=SIGMA)
        t1a, t1b, t2a, t2b = find_transit_slopes(light_curve, transit_start, transit_end, sigma=SIGMA)

        self.edges = {
            "transit_start": transit_start,
            "transit_end": transit_end,
            "t1a": t1a,
            "t1b": t1b,
            "t2a": t2a,
            "t2b": t2b
        }

    def set_edges(self, edges: dict | None) -> None:
        if edges:
            self.edges = edges
        else:
            self.edges = {
                "transit_start": None,
                "transit_end": None,
                "t1a": None,
                "t1b": None,
                "t2a": None,
                "t2b": None
            }

    def _get_edges(self) -> tuple[int, int, int, int]:
        edges = self.edges
        left_st = edges["t1a"] or edges["transit_start"] or edges["t1b"]
        left_en = edges["t1b"] or edges["transit_start"] or edges["t1a"]
        
        right_st = edges["t2a"] or edges["transit_end"] or edges["t2b"]
        right_en = edges["t2b"] or edges["transit_end"] or edges["t2a"]
        
        return left_st, left_en, right_st, right_en


class TransitData:
    def __init__(self, path: Path, planet_id: int, transit_num: int, axis_info: pd.DataFrame) -> None:
        self.planet_id = planet_id
        self.transit_num = transit_num
        self.airs = SensorData(path, planet_id, "AIRS-CH0", transit_num, axis_info=axis_info)
        self.fgs = SensorData(path, planet_id, "FGS1", transit_num)
        
    def process(self) -> None:
        self.airs.process()
        self.fgs.process()

        self.fgs.set_edges(
            self.correct_edges(
                self.airs.signal.shape[0],
                self.fgs.signal.shape[0],
                self.airs.edges
            )
        )
        self._calc_static_component()

    def correct_edges(self, airs_max_index: int, fgs_max_index: int, edges: dict) -> None:
        coef = fgs_max_index / airs_max_index
        edges = edges.copy()
        
        for k, v in edges.items():
            if v:
                edges[k] = int(v * coef)
        return edges
    
    def _calc_static_component(self) -> None:
        cfg = StaticModelConfig()
        cfg.MODEL_PHASE_DETECTION_SLICE = slice(10, len(self.fgs.signal) - 10)
        model = StaticModel(cfg)

        fgs = self.fgs.signal[:, 10:22, 10:22].reshape(self.fgs.signal.shape[0], -1)
        airs = self.airs.signal[:, 10:22, 39:321]

        pdata = np.concatenate([
            airs.mean(axis=1),
            fgs.mean(axis=1)[:, None],
        ], axis=1)

        self.static_component = model.predict_static(pdata)
        # self.spectre = model.predict_spectre(pdata)

    def get_map(self, channel_wise_normalization: bool = True) -> np.ndarray:
        left, _, _, right = self.airs._get_edges()

        airs = np.nan_to_num(self.airs.signal).astype(np.float32)[:, :, 39:].mean(axis=1)
        fgs = np.nan_to_num(self.fgs.signal).astype(np.float32).mean(axis=(1, 2))
        signal = np.concatenate([airs, fgs[:, None]], axis=1)
        
        star = np.concatenate([signal[:left], signal[right:]])

        if channel_wise_normalization:
            mean = star.mean(axis=0)[None, :]
        else:
            mean = star.mean()

        signal = signal / mean
        return signal

    # def get_map(self, channel_wise_normalization: bool = True) -> np.ndarray:
    #     left, _, _, right = self.airs._get_edges()

    #     airs = np.nan_to_num(self.airs.signal).astype(np.float32)
    #     fgs = np.nan_to_num(self.fgs.signal).astype(np.float32)
    #     l = airs.shape[0]

    #     signal = np.concatenate([
    #         airs[:, 14:18, 29:].reshape(l*4, 327),
    #         fgs[:, 14:18, 14:18].reshape(l*4, 4)
    #     ], axis=-1)
        
    #     star = np.concatenate([signal[:left], signal[right:]])

    #     if channel_wise_normalization:
    #         mean = star.mean(axis=0)[None, :]
    #     else:
    #         mean = star.mean()

    #     signal = signal / mean
    #     return signal
        

class DataProcessor:
    def __init__(self, planets: list[Path], axis_info: pd.DataFrame, cache_folder: str | None = None) -> None:
        self.planets = planets
        self.axis_info = axis_info

        if cache_folder is not None:
            self.cache_folder = Path(cache_folder) / "objects" 
            self.cache_folder.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_folder = None

        self.process()

    def process(self) -> None:
        self.data = []

        for planet_path in self.planets:
            files = planet_path.glob('*.parquet')
            num_transits = len(list(files)) // 2

            for i in range(num_transits):
                row = {
                    'path': planet_path.parent,
                    'planet_id': planet_path.name,
                    'num_transit': i
                }
                self.data.append(row)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> TransitData:
        row = self.data[index]

        if self.cache_folder:
            cache_file = self.cache_folder / f"{row['planet_id']}_{row['num_transit']}.joblib"
            if cache_file.exists():
                return joblib.load(cache_file)
                
        path = row['path']
        planet_id = row['planet_id']
        transit_num = row['num_transit']
        obj = TransitData(path, planet_id, transit_num, self.axis_info)
        obj.process()

        if self.cache_folder:
            joblib.dump(obj, cache_file)

        return obj
  

STATS = {
    "AIRS-CH0": [36892, 122805],
    "FGS1": [99257, 684636]
}

META_COLUMNS = ['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']
META_STATS = {
    'Rs': [1.25, 0.35],
    'Ms': [1.09, 0.21],
    'Ts': [5839.47, 511.61],
    'Mp': [1.57, 2.26],
    'P': [5.19, 1.41],
    'sma': [10.83, 3.03],
    'i': [88.46, 0.98]
    }
    
class TransitDataset(Dataset):
    def __init__(
            self, 
            data_processor: DataProcessor, 
            gt: pd.DataFrame | None, 
            meta: pd.DataFrame,
            output_stats: dict | None = None,
            transit_len: int = 40,
            augmentations: AugmentationList | None = None,
            ) -> None:
        self.cache = {}
        self.transit_len = transit_len
        self.data_processor = data_processor
        self.augmentations = augmentations if augmentations is not None else AugmentationList([])

        if gt is not None:
            self.gt = gt.set_index("planet_id").to_dict("index")
        else:
            self.gt = None

        self.meta = meta.set_index("planet_id").to_dict("index")
        self.cut_inf = 39
        self.cut_sup = 321
        
        self.output_stats = None
        self.precalc = True
        if output_stats is None:
            self.output_stats = self._precalc_stats()
        else:
            self.output_stats = output_stats

        self.precalc = False

    def __len__(self) -> int:
        return len(self.data_processor)
    
    def _process(self, a: np.ndarray, sensor: str) -> np.ndarray:
        diff = torch.from_numpy(np.nan_to_num(a))
        mean, std = STATS[sensor]
        diff = (diff - mean) / std
        return diff.to(torch.float32)
    
    def _signal_process(self, signal: np.ndarray, sensor: str, indices: list[np.ndarray]) -> np.ndarray:
        output = []
        for ix in indices:
            output.append(torch.stack([self._process(signal[i], sensor) for i in ix]))

        return torch.stack(output)
    
    def _meta_process(self, planet_id: int) -> np.ndarray:
        res = []
        for col in META_COLUMNS:
            val = self.meta[planet_id][col]
            mean, std = META_STATS[col]
            val = (val - mean) / std

            if len(self.augmentations) > 0 and random.random() < 0.5:
                val += np.random.randn() * std * 0.01

            res.append(val)

        return np.array(res).astype(np.float32)
    
    def _get_edges(self, edges: dict) -> tuple[int, int, int, int]:
        left_st = edges["t1a"] or edges["transit_start"] or edges["t1b"]
        left_en = edges["t1b"] or edges["transit_start"] or edges["t1a"]
        
        right_st = edges["t2a"] or edges["transit_end"] or edges["t2b"]
        right_en = edges["t2b"] or edges["transit_end"] or edges["t2a"]
        
        return left_st, left_en, right_st, right_en
    
    def _get_white_curve(self, signal: np.ndarray) -> np.ndarray:
        return signal.mean(axis=1)
    
    def _get_transit_map(self, obj: TransitData) -> np.ndarray:
        # return np.clip(obj.get_map(channel_wise_normalization=True), None, 1.5)[None, ...]
        return obj.get_map(channel_wise_normalization=True)[None, ...]
    
    def _add_meta_layer(self, transit_map: np.ndarray, meta: np.ndarray) -> np.ndarray:
        meta_layer = np.zeros_like(transit_map)
        feature_width = int(meta_layer.shape[-1] / meta.shape[0])
        for i in range(meta.shape[0]):
            meta_layer[0, :, i*feature_width:(i+1)*feature_width] = meta[i]

        return np.concatenate([transit_map, meta_layer], axis=0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index not in self.cache:
            obj = self.data_processor[index]
            planet_id = int(obj.planet_id)

            if self.gt is not None:
                targets = np.array([self.gt[planet_id][f"wl_{i}"] for i in range(1, 284)]).astype(np.float32)
            else:
                targets = None

            # white_curve = self._get_white_curve(signal)
            transit_map = self._get_transit_map(obj)

            meta = self._meta_process(planet_id)
            # transit_map = self._add_meta_layer(transit_map, meta)

            output = {
                # "white_curve": white_curve[None, :],
                "transit_map": transit_map,
                "meta": torch.from_numpy(meta),
                "planet_id": str(planet_id),
                "static_component": torch.from_numpy(np.array([obj.static_component]).astype(np.float32)),
                # "spectre": savgol_filter(obj.spectre.astype(np.float32), 18, 2)
            }

            if targets is not None:
                output["targets"] = torch.from_numpy(targets)

            self.cache[index] = output

        cached_output = self.cache[index]
        output = {
            "transit_map": cached_output["transit_map"].copy(),
            "meta": cached_output["meta"].clone(),
            "planet_id": cached_output["planet_id"],
            "static_component": cached_output["static_component"].clone(),
        }

        if "targets" in cached_output:
            output["targets"] = cached_output["targets"].clone()
        
        if not self.precalc:
            output["transit_map"] = torch.from_numpy(
                self.augmentations(output["transit_map"])
                )
            
            if len(self.augmentations) > 0 and random.random() < 0.5:
                # add random noise
                output["static_component"] += torch.randn_like(output["static_component"]) * 0.0005

        else:
            output["transit_map"] = torch.from_numpy(output["transit_map"])
            
        output = self._postprocess(output)

        return output
    
    

    def _postprocess(self, output: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        for k in output:
            if k in ["planet_id", "start", "end", "samples"]:
                continue

            if self.output_stats is not None:
                mean = self.output_stats[k]["mean"]
                std = self.output_stats[k]["std"]

                output[k] = (output[k] - mean) / std
            
        return output
    
    def _precalc_stats(self) -> dict[str, dict[str, float]]:
        stats = {}
        output_sample = self.__getitem__(0)
        
        for k in output_sample:
            if k in ["planet_id", "start", "end", "samples"]:
                continue
            
            arr = []
            num_samples = min(500, self.__len__())
            for i in tqdm(range(num_samples), total=num_samples, desc=f"Calculating stats for {k}"):
                sample = self.__getitem__(i)
                arr.append(sample[k])
            
            arr = torch.stack(arr, dim=0)
            
            if k == "transit_map":
                stats[k] = {"mean": torch.mean(arr, dim=(0, 2, 3))[:, None, None], "std": torch.std(arr, dim=(0, 2, 3))[:, None, None]}
            elif k in ["static_component", "targets"]:
                stats[k] = {"mean": torch.mean(arr), "std": torch.std(arr)}
            elif k == "meta":
                stats[k] = {"mean": torch.mean(arr, dim=0), "std": torch.std(arr, dim=0)}
            

        return stats

    def denorm(self, x: torch.Tensor, key: str) -> torch.Tensor:
        return x * self.output_stats[key]["std"].to(x.device) + self.output_stats[key]["mean"].to(x.device)
    
    def get_stats(self) -> dict[str, dict[str, float]]:
        return self.output_stats
    

# datamodule

class TransitDataModule(L.LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 16,
            num_workers: int = 0,
            test_size: float = 0.2,
            random_state: int = 42,
            full_train: bool = False,
            fold: int = 0
            ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.full_train = full_train
        self.fold = fold

        self.planets_stop_list = [
            "926530491",
            "905997089",
            "1124834224",
            "806204363",
            "158006264",
            "158006264",
            "561423413",
            "1843015807",
            "1338107575"
        ]


    def setup(self, stage: str | None = None, load_stats: bool = False) -> None:
        planets = sorted(list((self.data_path / "train").glob("*")))

        # if not self.full_train:
        #     train_planets, test_planets = train_test_split(
        #         planets, test_size=self.test_size, random_state=self.random_state
        #     )
        # else:
        #     train_planets, test_planets = planets, planets

        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        train_indices, test_indices = list(kfold.split(planets))[self.fold]
        train_planets = [planets[i] for i in train_indices]
        test_planets = [planets[i] for i in test_indices]

        print("before", len(train_planets), len(test_planets))

        train_planets = [p for p in train_planets if p.name not in self.planets_stop_list]
        test_planets = [p for p in test_planets if p.name not in self.planets_stop_list]
        
        print("after", len(train_planets), len(test_planets))

        planets_gt = pd.read_csv(self.data_path / "train.csv")
        meta = pd.read_csv(self.data_path / "train_star_info.csv")
        axis_info = pd.read_parquet(self.data_path / "axis_info.parquet")

        train_processor = DataProcessor(train_planets, axis_info=axis_info, cache_folder="cached_data")
        test_processor = DataProcessor(test_planets, axis_info=axis_info, cache_folder="cached_data")

        stats_path = Path(f"stats_{self.fold}.joblib")
        if load_stats and stats_path.exists():
            output_stats = joblib.load(stats_path)
        else:
            output_stats = None

        self.train_dataset = TransitDataset(train_processor, planets_gt, meta, output_stats=output_stats, augmentations=get_augmentations("train"))
        joblib.dump(self.train_dataset.get_stats(), stats_path)

        self.val_dataset = TransitDataset(test_processor, planets_gt, meta, output_stats=self.train_dataset.get_stats(), augmentations=get_augmentations("val"))


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )