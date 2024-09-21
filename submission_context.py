import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class SubmissionContext(BaseModel):
    temp_dir: str
    model: Any = None
    results: Dict[str, float] = None
    env_id: str = Field(default="CartPole-v1", alias="ENV_NAME")
    outputs_dir: str = Field(default="outputs", alias="OUTPUTS_DIR")
    video_name: str = Field(default="model_evaluation", alias="STUDY_NAME")
    model_dir: str = Field(default="models", alias="MODEL_DIR")
    best_model_name: str = Field(default="best_model.zip", alias="BEST_MODEL_NAME")
    model_architecture: str = "PPO"
    n_eval_episodes: int = Field(default=10, alias="N_EVAL_EPISODES")
    hf_token: str = None
    hf_username: str = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_card: Optional[str] = None

    def get_model_path(self) -> str:
        """Get the full path to the best model file."""
        return os.path.join(self.model_dir, self.best_model_name)

    def get_results_path(self) -> str:
        """Get the full path to the results JSON file."""
        return os.path.join(self.outputs_dir, "results.json")

    def get_video_dir(self) -> str:
        """Get the directory containing video files."""
        return os.path.join(self.outputs_dir, "videos")

    def get_repo_id(self) -> str:
        """Get the repository ID for Hugging Face Hub."""
        return f"{self.hf_username}/{self.env_id}-{self.model_architecture}"

    @classmethod
    def from_env(cls, temp_dir: str):
        load_dotenv()
        return cls(
            temp_dir=temp_dir,
            hf_token=os.getenv("HF_TOKEN"),
            hf_username=os.getenv("HF_USERNAME"),
        )

    class Config:
        populate_by_name = True
        protected_namespaces = ()
