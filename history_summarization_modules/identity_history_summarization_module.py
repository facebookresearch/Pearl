from pearl.api.history import History
from pearl.api.observation import Observation
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
    SubjectiveState,
)


class IdentityHistorySummarizationModule(HistorySummarizationModule):
    """
    A history summarization module that simply uses the original observations.
    """

    def __init__(self) -> None:
        super(IdentityHistorySummarizationModule, self).__init__()
        self.history: History = None

    def summarize_history(self, observation: Observation) -> SubjectiveState:
        self.history = observation
        return observation

    def get_history(self) -> History:
        return self.history

    def forward(self, x: History) -> SubjectiveState:
        return x
