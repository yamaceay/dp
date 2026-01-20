from dp.methods._dpmlm import DPMlmAnonymizer


class DPMlmLongformerAnonymizer(DPMlmAnonymizer):
    MODEL_NAME = "dpmlm_longformer"

    def __init__(
        self,
        *args,
        model_checkpoint: str = "allenai/longformer-base-4096",
        max_length: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            model_checkpoint=model_checkpoint,
            max_length=max_length,
            **kwargs,
        )
