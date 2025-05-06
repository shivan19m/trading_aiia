from abc import ABC, abstractmethod
from utils.helpers import validate_plan_structure
import random
import logging
from openai import OpenAI
from openai.types.chat import ChatCompletion
import time
    
class BaseAgent(ABC):
    """
    Abstract base class for all agents in the financial planning system.
    """
    def __init__(self):
        self.tickers = None  # Will be set by the system
    
    def set_tickers(self, tickers):
        """Set the list of valid tickers for this agent."""
        self.tickers = tickers
    
    def validate_plan(self, plan):
        """
        Validate and normalize a trading plan structure.
        Ensures each ticker and 'cash' has a dict with at least a 'weight' key.
        """
        if self.tickers is None:
            raise ValueError("Tickers not set for agent. Call set_tickers() first.")
        return validate_plan_structure(plan, self.tickers)



    logger = logging.getLogger(__name__)

    @staticmethod
    def call_openai_with_backoff(
        client: OpenAI,
        model: str,
        messages: list,
        temperature: float = 0.3,
        max_tokens: int = 500,
        max_retries: int = 5
    ) -> ChatCompletion:
        """
        Static method to call OpenAI API with exponential backoff on rate limiting.
        """
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    BaseAgent.logger.warning(
                        f"[OpenAI] Rate limited. Retry {attempt+1}/{max_retries} after {wait:.2f}s"
                    )
                    time.sleep(wait)
                else:
                    BaseAgent.logger.error(f"[OpenAI] API call failed: {e}")
                    raise
        raise RuntimeError("Exceeded max retries due to rate limiting.")

    @abstractmethod
    def propose_plan(self, market_data, context):
        """
        Generate a trading plan based on market data and context.
        Returns a validated plan dict with weights for each ticker and cash.
        """
        raise NotImplementedError("propose_plan() must be implemented by subclass.")

    @abstractmethod
    def justify_plan(self, plan, context):
        """
        Provide justification for the proposed plan.
        TODO: Integrate LLM-based justification.
        """
        return None

    @abstractmethod
    def critique_plan(self, plan, context):
        """
        Critique another agent's plan.
        TODO: Implement Socratic reasoning and critique logic.
        """
        return None

    @abstractmethod
    def validate_constraints(self, plan, constraints):
        """
        Validate the plan against explicit, implicit, and derived constraints.
        TODO: Implement constraint validation logic.
        """
        return True, []

    @abstractmethod
    def execute(self, plan):
        """
        Execute the plan (or simulate execution).
        TODO: Integrate with ExecutorAgent and real execution logic.
        """
        return {'status': 'skipped', 'plan': plan} 
    
    