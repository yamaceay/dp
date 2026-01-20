from typing import Optional

from dp.tri.with_bk import TRIDetectorWithBK


class TRIDetectorWithBKLongformer(TRIDetectorWithBK):
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        model_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,
        device: str = "auto",
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            model_name=model_name,
            max_length=max_length,
            device=device,
            use_chunking=False,
            use_overflow=False,
        )
