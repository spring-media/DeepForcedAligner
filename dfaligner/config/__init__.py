from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional

from everyvoice.config.preprocessing_config import PreprocessingConfig
from everyvoice.config.shared_types import (
    AdamOptimizer,
    AdamWOptimizer,
    BaseModelWithContact,
    BaseTrainingConfig,
    ConfigModel,
    init_context,
)
from everyvoice.config.text_config import TextConfig
from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.config.utils import load_partials
from everyvoice.utils import load_config_from_json_or_yaml_path
from pydantic import Field, FilePath, ValidationInfo, field_serializer, model_validator

# DFAlignerConfig's latest version number
LATEST_VERSION: str = "1.0"


class DFAlignerExtractionMethod(Enum):
    beam = "beam"
    dijkstra = "dijkstra"


class DFAlignerModelConfig(ConfigModel):
    target_text_representation_level: TargetTrainingTextRepresentationLevel = (
        TargetTrainingTextRepresentationLevel.characters
    )
    lstm_dim: int = Field(
        512, description="The number of dimensions in the LSTM layers."
    )
    conv_dim: int = Field(
        512, description="The number of dimensions in the convolutional layers."
    )

    @field_serializer("target_text_representation_level")
    def convert_training_enum(
        self, target_text_representation_level: TargetTrainingTextRepresentationLevel
    ):
        return target_text_representation_level.value


class DFAlignerTrainingConfig(BaseTrainingConfig):
    optimizer: AdamOptimizer | AdamWOptimizer = Field(
        default_factory=AdamWOptimizer,
        description="Optimizer configuration settings.",
    )
    binned_sampler: bool = Field(True, description="Use a binned length sampler")
    plot_steps: int = Field(1000, description="The maximum number of steps to plot")
    extraction_method: DFAlignerExtractionMethod = Field(
        DFAlignerExtractionMethod.dijkstra,
        description="The alignment extraction algorithm to use. 'beam' will be quicker but possibly less accurate than 'dijkstra'",
    )

    @field_serializer("extraction_method")
    def convert_extraction_method_enum(
        self, extraction_method: DFAlignerExtractionMethod
    ):
        return extraction_method.value


class DFAlignerConfig(BaseModelWithContact):
    VERSION: Annotated[
        str,
        Field(
            default_factory=lambda: LATEST_VERSION,
            init_var=False,
        ),
    ]

    # TODO FastSpeech2Config and DFAlignerConfig are almost identical.
    model: DFAlignerModelConfig = Field(
        default_factory=DFAlignerModelConfig,
        description="The model configuration settings.",
    )
    path_to_model_config_file: Optional[FilePath] = Field(
        None, description="The path of a preprocessing configuration file."
    )

    training: DFAlignerTrainingConfig = Field(
        default_factory=DFAlignerTrainingConfig,
        description="The training configuration hyperparameters.",
    )
    path_to_training_config_file: Optional[FilePath] = Field(
        None, description="The path of a preprocessing configuration file."
    )

    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="The preprocessing configuration, including information about audio settings.",
    )
    path_to_preprocessing_config_file: Optional[FilePath] = Field(
        None, description="The path of a preprocessing configuration file."
    )

    text: TextConfig = Field(default_factory=TextConfig)
    path_to_text_config_file: Optional[FilePath] = None

    @model_validator(mode="before")  # type: ignore
    def load_partials(self: dict[Any, Any], info: ValidationInfo):
        config_path = (
            info.context.get("config_path", None) if info.context is not None else None
        )
        return load_partials(
            self,
            ("model", "training", "preprocessing", "text"),
            config_path=config_path,
        )

    @staticmethod
    def load_config_from_path(path: Path) -> "DFAlignerConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = DFAlignerConfig(**config)
        return config

    @model_validator(mode="before")
    @classmethod
    def check_and_upgrade_checkpoint(cls, data: Any) -> Any:
        """
        Check model's compatibility and possibly upgrade.
        """
        ckpt_version = data.get("VERSION", "0.0")
        if ckpt_version > LATEST_VERSION:
            raise ValueError(
                "Your config was created with a newer version of EveryVoice, please update your software."
            )
        # Successively convert model checkpoints to newer version.
        if ckpt_version < "1.0":
            # TODO: Write code to convert model to version 1.0.
            data["VERSION"] = "1.0"

        return data

    # INPUT_TODO: initialize text with union of symbols from dataset
