from pathlib import Path
import itertools

import pandas as pd
import numpy as np
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib


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
            self.bin_coef = 30
        else:
            self.bin_coef = 360
        
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
        self.signal = np.nan_to_num(self.signal)

    def process(self) -> None:
        self.apply_adc_convert()
        self.apply_mask_hot_dead()
        self.apply_linear_corr()

        self.apply_clean_dark()
        self.apply_clean_read()

        self.apply_cds()
        self.find_edges()
        self.bin_obs(self.bin_coef)

        self.apply_correct_flat_field()
        # self.fill_nan()

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

    def plot_raw(self, time: int) -> plt.Figure:
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
        t1a, t1b, t2a, t2b = find_transit_slopes(light_curve, transit_start, transit_end, sigma=60)

        self.edges = {
            "transit_start": transit_start,
            "transit_end": transit_end,
            "t1a": t1a,
            "t1b": t1b,
            "t2a": t2a,
            "t2b": t2b
        }

   

class TransitData:
    def __init__(self, path: Path, planet_id: int, transit_num: int, axis_info: pd.DataFrame) -> None:
        self.planet_id = planet_id
        self.transit_num = transit_num
        self.airs = SensorData(path, planet_id, "AIRS-CH0", transit_num, axis_info=axis_info)
        self.fgs = SensorData(path, planet_id, "FGS1", transit_num)
        
    def process(self) -> None:
        self.airs.process()
        self.fgs.process()

class DataProcessor:
    def __init__(self, planets: list[Path], axis_info: pd.DataFrame) -> None:
        self.planets = planets
        self.axis_info = axis_info

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
        path = row['path']
        planet_id = row['planet_id']
        transit_num = row['num_transit']

        return TransitData(path, planet_id, transit_num, self.axis_info)
    
    def save(self, path: Path | str, plots: bool = False) -> None:    
        folder = Path(path)
        object_folder = folder / "objects"
        object_folder.mkdir(parents=True, exist_ok=True)

        if plots:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            plt.style.use('ggplot')

            plots_folder = folder / "plots"

            airs_raw_plot_folder = plots_folder / "airs_raw_plots"
            airs_curve_plot_folder = plots_folder / "airs_curve_plots"

            fgs_raw_plot_folder = plots_folder / "fgs_raw_plots"
            fgs_curve_plot_folder = plots_folder / "fgs_curve_plots"

            plots_folder.mkdir(parents=True, exist_ok=True)
            airs_raw_plot_folder.mkdir(parents=True, exist_ok=True)
            airs_curve_plot_folder.mkdir(parents=True, exist_ok=True)
            fgs_raw_plot_folder.mkdir(parents=True, exist_ok=True)
            fgs_curve_plot_folder.mkdir(parents=True, exist_ok=True)


        for i, d in tqdm(enumerate(self), desc="Processing data", total=len(self)):
            try:
                d.process()
                name = f"{d.planet_id}_{d.transit_num}"
                joblib.dump(d, object_folder / f"{name}.joblib")

                raw = d.airs.plot_raw(10)
                raw.savefig(airs_raw_plot_folder / f"{name}.png", bbox_inches="tight")

                curve = d.airs.plot_curve()
                curve.savefig(airs_curve_plot_folder / f"{name}.png", bbox_inches="tight")

                raw = d.fgs.plot_raw(10)
                raw.savefig(fgs_raw_plot_folder / f"{name}.png", bbox_inches="tight")

                curve = d.fgs.plot_curve()
                curve.savefig(fgs_curve_plot_folder / f"{name}.png", bbox_inches="tight")
            except Exception as e:
                print(f"error in object {i}", e)
                continue

    