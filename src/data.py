from pathlib import Path
import itertools

import pandas as pd
import numpy as np
from astropy.stats import sigma_clip


sensor_sizes_dict = {
    "AIRS-CH0": [[11250, 32, 356], [32, 356]],
    "FGS1": [[135000, 32, 32], [32, 32]],
}



class SensorData:
    def __init__(
            self, 
            path: Path | str, 
            planet_id: int, 
            sensor: str, 
            transit_num: int,
            axis_info: pd.DataFrame | None = None
            ) -> None:
        if sensor not in ["AIRS-CH0", "FGS1"]:
            raise ValueError("Invalid sensor")
        
        self.path = Path(path)
        self.sensor = sensor
        self.planet_id = planet_id
        self.transit_num = transit_num
        self.gain = 0.4369
        self.offset = -1000.0
        
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

    def adc_convert(self) -> None:
        if self.adc_converted:
            return
        
        self.signal /= self.gain
        self.signal += self.offset
        self.adc_converted = True

    def mask_hot_dead(self) -> None:
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

    def clean_dark(self) -> None:
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

    def correct_flat_field(self) -> None:
        if self.flat_field_converted:
            return
        
        self.signal = self.signal / self.flat_frame
        self.flat_field_converted = True

    def clean_read(self) -> None:
        if self.read_converted:
            return
        
        read = np.tile(self.read_frame * 2, (self.signal.shape[0], 1, 1))
        self.signal -= read
        self.read_converted = True

   

class TransitData:
    def __init__(self, path: Path, planet_id: int, transit_num: int, axis_info: pd.DataFrame) -> None:
        self.airs = SensorData(path, planet_id, "AIRS-CH0", transit_num, axis_info=axis_info)
        self.fgs = SensorData(path, planet_id, "FGS1", transit_num)
        


class DataProcessor:
    def __init__(self, path: str) -> None:
        self.path = path
        self.files = list(Path(path).glob("*"))

    