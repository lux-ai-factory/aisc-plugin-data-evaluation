import io
import zipfile
from collections.abc import Iterator
from datetime import datetime

import pandas as pd
from a4s_plugin_interface.input_providers.base_input_provider import BaseInputProvider

from .iterators import DateIterator


class DataFrameProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes | list[bytes]) -> dict[str, pd.DataFrame]:
        if isinstance(file_content, bytes):
            # Check if it's a zip file by signature
            if file_content.startswith(b"PK\x03\x04"):
                return self._read_zip_file(file_content)
            else:
                return {"test": self._read_single_file(file_content)}
        elif file_content[0].startswith(b"PK\x03\x04"):
            # if the model was provided, and the "test" set is a zip file,
            # we will discard the train set and assume refrence and evaluated are in the zip file
            return self._read_zip_file(file_content[0])

        return {
            name: self._read_single_file(f)
            for name, f in zip(("train", "test"), file_content)
        }

    def _read_zip_file(self, file_content: bytes) -> dict[str, pd.DataFrame]:
        import pandas as pd

        fnames = ("train.csv", "test.csv")

        files = {}
        with zipfile.ZipFile(io.BytesIO(file_content)) as z:
            for name in z.namelist():
                if name in fnames:
                    files[name] = z.read(name)

        assert len(files) == 2

        return {
            "train": pd.read_csv(io.BytesIO(files[fnames[0]])),
            "test": pd.read_csv(io.BytesIO(files[fnames[1]])),
        }

    def _read_single_file(self, file_content: bytes) -> pd.DataFrame:
        import pandas as pd

        file_stream = io.BytesIO(file_content)
        try:
            return pd.read_parquet(file_stream)
        except Exception:
            file_stream.seek(0)
            try:
                return pd.read_csv(file_stream)
            except Exception as e:
                raise ValueError("File is neither a valid Parquet nor CSV.") from e

    def iter(
        self,
        date_feature: str | None,
        frequency: str | None,
        window_size: str | None,
        date_round: str | None = "1 D",
    ) -> Iterator[tuple[datetime | None, pd.Series]]:
        if date_feature is not None:
            self._data["test"][date_feature] = pd.to_datetime(
                self._data["test"][date_feature]
            )

        yield from DateIterator(
            self._data["test"], date_feature, frequency, window_size, date_round
        )
