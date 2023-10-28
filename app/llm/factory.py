from typing import Dict, Tuple
import app.llm.prompter as prompter
import app.llm.models as models


class LLMFactory:
    def __init__(self) -> None:
        self.llm_models: Dict[str, Tuple(models.LLM, prompter.Prompter)] = {}

    def register_llm_model(self, model_name, llm_model: models.LLM, prompter: prompter.Prompter):
        self.llm_models[model_name] = (llm_model, prompter)

    def create_llm(self, model_name, dir_path=None) -> models.LLM:
        if model_name not in self.llm_models:
            raise ValueError(f"Unsupported LLM model: {model_name}")

        prompter = self.llm_models[model_name][1]()
        return self.llm_models[model_name][0](prompter)


llm_factory = LLMFactory()
llm_factory.register_llm_model("Mock", models.MockLLM, prompter.MockPrompter)
llm_factory.register_llm_model("Toonchat_v2.1", models.HuggingfaceLLM, prompter.ToonchatV21Prompter)
llm_factory.register_llm_model("Toonchat_v2.3", models.HuggingfaceLLM, prompter.ToonchatV23Prompter)
