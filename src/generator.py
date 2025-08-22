import numpy as np
import pandas as pd
import exosim.recipes as recipes
import h5py
from exosim.log import setLogLevel
import logging
from pathlib import Path
import json
import joblib
import traceback

from src.data import SensorData, TransitData


params_to_tags = {
    "Rs": ['<R unit="R_sun">', '</R>'],
    "Ms": ['<M unit="M_sun">', '</M>'],
    "Ts": ['<T unit="K">', '</T>'],
    "D": ['<D unit="pc">', '</D>'],
    
    "t0": ["<t0 unit='hour'>", '</t0>'],
    "P": ["<period unit='day'>", '</period>'],
    "sma": ['<sma>', '</sma>'],
    "i": ["<inc unit='deg'>", '</inc>'],
    "e": ['<ecc>', '</ecc>'],
    "method": ["<limb_darkening>", '</limb_darkening>'],
    "ld_coeffs": ["<limb_darkening_coefficients>", '</limb_darkening_coefficients>'],
}


class TransitDataGenerator:
    def __init__(self, params: dict, wavelengths: np.ndarray) -> None:
        self.wavelengths = wavelengths
        self.params = params
       

    def generate(self) -> dict[str, np.ndarray]:
        self._write_target_file()
        self._create_sky_file()

        setLogLevel(logging.CRITICAL)

        recipes.CreateFocalPlane("main_example.xml", "output.h5")
        recipes.CreateSubExposures(
            input_file='output.h5',
            output_file="exp.h5",
            options_file="main_example.xml",
        )
        recipes.CreateNDRs(
            input_file="exp.h5",
            output_file="ndr_exp.h5",
            options_file="main_example.xml",
        )
        
        f = h5py.File('ndr_exp.h5', 'r')
        output = {
            "photometer": f["channels"]["Photometer"]["NDRs"]["data"][()],
            "spectrometer": f["channels"]["Spectrometer"]["NDRs"]["data"][()][:, :, ::-1],
        }
        f.close()

        # f = h5py.File('exp.h5', 'r')
        # output = {
        #     "photometer": f["channels"]["Photometer"]["SubExposures"]["data"][()],
        #     "spectrometer": f["channels"]["Spectrometer"]["SubExposures"]["data"][()][:, :, ::-1],
        # }
        # f.close()
        
        return output

    def _create_sky_file(self) -> None:
        xml_file = open("sky_init.xml", "r").readlines()
        for p, (start_tag, end_tag) in params_to_tags.items():
            
            is_set = False
            for i in range(len(xml_file)):
                line = xml_file[i]
                if start_tag in line:
                    start_pos = line.find(start_tag) + len(start_tag)
                    end_pos = line.find(end_tag)
                    
                    new_line = line[:start_pos] + str(self.params[p]) + line[end_pos:]
                    xml_file[i] = new_line
                    is_set = True
                    break

            if not is_set:
                raise ValueError(f"No tag found for parameter {p}")
                
            
        with open("sky_example.xml", "w") as f:
            f.writelines(xml_file)


    def _write_target_file(self) -> None:
        header = """
# %ECSV 1.0
# ---
# datatype:
# - {name: Wavelength, unit: um, datatype: float64}
# - {name: rp/rs, datatype: float64}
# schema: astropy-2.0
Wavelength rp/rs
"""

        content = [header.strip()]
        target = [self.params[f"wl_{i}"] for i in range(1, 284)]

        ins_wave = np.linspace(self.wavelengths[0], self.wavelengths[1], 150)
        ins_target = np.linspace(target[0], target[1], 150)

        upd_wave = np.concatenate([[self.wavelengths[0]], ins_wave, self.wavelengths[1:]])
        upd_target = np.concatenate([[target[0]], ins_target, target[1:]])

        for i in range(len(upd_wave) -1, -1, -1):
            content.append(f"{float(upd_wave[i])} {float(upd_target[i]) ** 0.5}")

        with open("planet_wl.csv", "w") as f:
            f.write("\n".join(content))


class GeneratedSensorData(SensorData):
    def __init__(
            self, 
            arrays: dict[str, np.ndarray],
            planet_id: int | str, 
            sensor: str
            ) -> None:
        if sensor not in ["AIRS-CH0", "FGS1"]:
            raise ValueError("Invalid sensor")
        
        if sensor == "AIRS-CH0":
            self.bin_coef = 20
        else:
            self.bin_coef = 23
        self.signal_len = 187
        
        self.sensor = sensor
        self.planet_id = planet_id
        self.signal = arrays["signal"]

        self._construct_dt()

        self.edges = {}
        self.dark_frame = arrays["dark_frame"].copy()
        self.dead_frame = arrays["dead_frame"].copy()
        self.read_frame = arrays["read_frame"].copy()
        self.linear_corr_frame = arrays["linear_corr_frame"].copy()

        self.adc_converted = False
        self.mask_hot_dead_converted = False
        self.linear_corr_converted = False
        self.clean_dark_converted = False
        self.cds_converted = False
        self.flat_field_converted = False
        self.read_converted = False
        self.binned = False
        self.interpolated = False

    def _construct_dt(self) -> None:
        dt = np.ones(len(self.signal)) * 0.1
        dt[1::2] += 0.1
        self.dt = dt

    def process(self) -> None:
        self.apply_mask_hot_dead()
        # self.apply_linear_corr()

        # self.apply_clean_dark()
        # self.apply_clean_read()

        self.apply_cds()

        if self.sensor == "AIRS-CH0":
            self.find_edges()
        else:
            self.set_edges(None)
        
        self.bin_obs(self.bin_coef)
        self.fill_nan()
        self.correct_signal_len()

    def correct_signal_len(self) -> None:
        self.signal = self.signal[:self.signal_len]

        if self.edges:
            for k, v in self.edges.items():
                if v and v >= self.signal_len:
                    self.edges[k] = None


class GeneratedTransitData(TransitData):
    def __init__(
            self, 
            arrays: dict[str, np.ndarray],
            airs_maps: dict[str, np.ndarray], 
            fgs_maps: dict[str, np.ndarray],
            planet_id: int | str
            ) -> None:
        self.planet_id = planet_id

        fgs_arr = {
            "signal": arrays["photometer"].copy()
        }
        fgs_arr.update(fgs_maps)
        self.fgs = GeneratedSensorData(fgs_arr, planet_id, "FGS1")

        airs_arr = {
            "signal": arrays["spectrometer"].copy()
        }
        airs_arr.update(airs_maps)
        self.airs = GeneratedSensorData(airs_arr, planet_id, "AIRS-CH0")



class GeneratedDataProcessor:
    def __init__(self, df: pd.DataFrame, cache_folder: str, wavelengths: pd.DataFrame) -> None:
        self.df = df

        self.bad_id_filename = Path("bad_ids.json")

        if self.bad_id_filename.exists():
            with open(self.bad_id_filename, "r") as f:
                self.bad_ids = json.load(f)

            self.df = df[~df["planet_id"].isin(self.bad_ids)]
        else:
            self.bad_ids = []

        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)

        self.airs_maps = {
            "dark_frame": np.load("NDR/airs/dark_map.npy"),
            "dead_frame": np.load("NDR/airs/dead_pixel_map.npy"),
            "read_frame": np.load("NDR/airs/read_noise_map.npy"),
            "linear_corr_frame": np.load("NDR/airs/pnl_map.npy"),
        }

        self.fgs_maps = {
            "dark_frame": np.load("NDR/fgs/dark_map.npy"),
            "dead_frame": np.load("NDR/fgs/dead_pixel_map.npy"),
            "read_frame": np.load("NDR/fgs/read_noise_map.npy"),
            "linear_corr_frame": np.load("NDR/fgs/pnl_map.npy"),
        }
        self.wavelengths = wavelengths.iloc[0].values

    def __len__(self) -> int:
        return len(self.df)
    
    def add_bad_id(self, planet_id: int | str) -> None:
        self.bad_ids.append(planet_id)
        with open(self.bad_id_filename, "w") as f:
            json.dump(self.bad_ids, f)
    
    def __getitem__(self, i: int) -> GeneratedTransitData | None:
        row = self.df.iloc[i].to_dict()
        
        cache_file = self.cache_folder / f"{row['planet_id']}.joblib"
        if cache_file.exists():
            gt = joblib.load(cache_file)
        
        else:
            try:
                gen = TransitDataGenerator(row, self.wavelengths)
                arrays = gen.generate()

                gt = GeneratedTransitData(arrays, fgs_maps=self.fgs_maps, airs_maps=self.airs_maps, planet_id=row["planet_id"])
                gt.process()
                joblib.dump(gt, cache_file)
            
            except Exception as e:
                self.add_bad_id(row["planet_id"])
                traceback.print_exc()
                return None
            
        return gt
