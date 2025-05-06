import os
from dotenv import load_dotenv
load_dotenv()  # Ensure .env is loaded for API key
import openai
import json
from agents.base import BaseAgent
import random
import re
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_json_from_response(response):
    # Remove code fences if present
    response = re.sub(r'```(?:json)?', '', response, flags=re.IGNORECASE).strip()
    # Find the first {...} block
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            logger.error(f"JSON parse error: {e}")
            return None
    return None

class MomentumAgent(BaseAgent):
    """
    Agent that generates trade plans based on price acceleration and volume trends using GPT-4.
    """
    def __init__(self, window: int = 5):
        super().__init__()  # Initialize base class
        self.window = window
        
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.tickers = []

    def set_tickers(self, tickers: list):
        """Set the list of tickers to consider."""
        self.tickers = tickers

    def apply_rule_fallback(self, features: Dict[str, Any], current_holdings, cash) -> Dict[str, Dict[str, Any]]:
        """
        UPDATED: rule-based fallback that allocates proportionally to 5-day momentum,
        preserves existing winners, and leaves some cash buffer.
        """
        scores = {}
        for sym, vals in features.items():
            if sym.lower() == "cash" or not isinstance(vals, dict):
                continue
            # simple score = recent pct_change
            scores[sym] = vals.get("momentum", 0.0)

        # only positive momentum
        pos = {s: max(0, sc) for s, sc in scores.items()}
        total = sum(pos.values())
        plan = {}

        if total > 0:
            # allocate 80% proportionally to positive momentum
            for sym, sc in pos.items():
                plan[sym] = {
                    "weight": 0.8 * (sc / total),
                    "reason": "Rule: proportional to 5-day momentum"
                }
            # preserve current winners by bumping their weight if already held
            for sym, held_w in (current_holdings or {}).items():
                if sym in plan and held_w > plan[sym]["weight"]:
                    plan[sym]["weight"] = held_w
            # cash buffer
            plan["cash"] = {"weight": 0.2, "reason": "Rule: cash buffer"}
        else:
            # no signal → equal weight across all tickers
            n = len(self.tickers)
            w = 1.0 / n if n else 1.0
            for sym in self.tickers:
                plan[sym] = {
                    "weight": w,
                    "reason": "Fallback: equal‑weight allocation"
                }

        return plan

    def propose_plan(
        self,
        features: Dict[str, Any],
        context: str,
        memory_agent=None,
        current_holdings: Dict[str, float]=None,
        cash: float=None,
        portfolio_history: list=None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate a portfolio allocation plan based on momentum indicators and current portfolio state.
        """

        if not self.tickers:
            logger.error("No tickers set for MomentumAgent.")
            raise ValueError("Tickers must be set before calling propose_plan.")

        # 1) prepare individual feature dict
        feat = {t: features.get(t, {}) for t in self.tickers}
        # 2) stringify portfolio state
        holdings_str = json.dumps(current_holdings or {}, indent=2)
        cash_str     = f"{cash:.2f}" if cash is not None else "N/A"
        history_str  = json.dumps((portfolio_history or [])[-3:], indent=2)

        # 3) build prompt
        prompt = (
            "You are a momentum-based portfolio manager.\n"
            "Focus on 5-day price momentum and avoid unnecessary turnover.\n\n"
            f"Context: {context}\n\n"
            f"Features:\n{json.dumps(feat, indent=2)}\n\n"
            "Current Holdings:\n"
            f"{holdings_str}\n\n"
            f"Cash: {cash_str}\n\n"
            "Recent Portfolio History (last 3 periods):\n"
            f"{history_str}\n\n"
            "Return ONLY valid JSON mapping each symbol to {weight, reason}, "
            "weights must sum to 1.0, no markdown or explanation.\n"
        )

        try:
            # 4) LLM call with backoff
            resp = self.call_openai_with_backoff(
                client=self.client,
                model="gpt-3.5-turbo",         # use an available model
                messages=[{"role":"system","content":"Output valid JSON only."},
                          {"role":"user",  "content":prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            raw = resp.choices[0].message.content.strip()

            # 5) strip code fences if any
            clean = raw.replace("```json","").replace("```","").strip()

            # 6) parse JSON
            plan = json.loads(clean)

            # 7) validate and normalize
            total = 0.0
            for sym, info in plan.items():
                if not isinstance(info, dict):
                    raise ValueError(f"Bad info for {sym}")
                w = info.get("weight", 0.0)
                if not (0.0 <= w <= 1.0):
                    raise ValueError(f"Weight {w} for {sym} out of bounds")
                info.setdefault("reason", "No reason provided")
                total += w
            if abs(total - 1.0) > 1e-2:
                raise ValueError(f"Weights sum to {total:.2f}, not 1.0")

            return plan

        except Exception as e:
            # UPDATED: any failure → rule fallback
            logger.warning(f"[MomentumAgent] LLM failed, using rule fallback: {e}")
            return self.apply_rule_fallback(feat, current_holdings, cash)

        finally:
            # UPDATED: store plan in memory if succeeded
            if memory_agent and 'plan' in locals():
                memory_agent.store_plan_vector(plan, json.dumps(feat))
    # def propose_plan(self, features: Dict[str, Any], context: str, memory_agent=None, current_holdings=None, cash=None, portfolio_history=None) -> Dict[str, Dict[str, Any]]:
    #     """
    #     Generate a portfolio allocation plan based on momentum indicators and current portfolio state.
    #     """
    #     if not self.tickers:
    #         logger.error("No tickers set for MomentumAgent. Cannot generate plan.")
    #         raise ValueError("No tickers set for MomentumAgent.")

    #     try:
    #         # Prepare features for each ticker
    #         ticker_features = {}
    #         for ticker in self.tickers:
    #             if ticker in features:
    #                 ticker_features[ticker] = features[ticker]
    #             else:
    #                 logger.warning(f"Missing features for {ticker}")

    #         # Add portfolio state to prompt
    #         holdings_str = json.dumps(current_holdings, indent=2) if current_holdings else '{}'
    #         cash_str = f"{cash}" if cash is not None else 'N/A'
    #         history_str = json.dumps(portfolio_history[-3:], indent=2) if portfolio_history else '[]'

    #         prompt = f"""
    #         Based on the following technical indicators, current portfolio holdings, cash, and recent portfolio history, generate a portfolio allocation plan.
    #         Focus on momentum strategies—identify strong trends and allocate accordingly, but avoid unnecessary trades if already optimally allocated. Prefer to hold winners and minimize turnover.

    #         Context: {context}

    #         Features:
    #         {json.dumps(ticker_features, indent=2)}

    #         Current Holdings:
    #         {holdings_str}

    #         Cash: {cash_str}

    #         Recent Portfolio History (last 3 windows):
    #         {history_str}

    #         Return ONLY a valid JSON object mapping each symbol to a dict with keys: weight (0-1), reason (string). Do not include any explanation, markdown, or code block formatting.
    #         """

    #         print(f"[AGENT] Prompt to LLM:\n{prompt}")

    #         # Get GPT response
    #         response = self.client.chat.completions.create(
    #             model="gpt-4",
    #             messages=[{"role": "user", "content": prompt}],
    #             temperature=0.7,
    #             max_tokens=1000
    #         )
    #         gpt_response = response.choices[0].message.content

    #         # Robust JSON extraction
    #         plan = extract_json_from_response(gpt_response)
    #         if not plan:
    #             logger.error("Failed to extract valid JSON plan from LLM response")
    #             return {}

    #         # Validate plan structure
    #         if not self._validate_plan(plan):
    #             logger.error("Plan validation failed")
    #             return {}

    #         return plan

    #     except Exception as e:
    #         logger.error(f"Error in propose_plan: {str(e)}")
    #         return {}
        
    def extract_features(self, market_data_dict):
        features = {}
        for symbol, df in market_data_dict.items():
            try:
                df = df.copy()
                df['momentum'] = df['close'].pct_change(periods=5)
                df['rsi'] = df['close'].rolling(14).apply(self._calc_rsi)
                df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
                latest = df.dropna().iloc[-1]
                features[symbol] = {
                    'momentum': float(latest['momentum']),
                    'rsi': float(latest['rsi']),
                    'macd': float(latest['macd']),
                }
            except Exception as e:
                features[symbol] = {}
        return features
    
    def _validate_plan(self, plan: Dict[str, Dict[str, Any]]) -> bool:
        """Validate the structure of the generated plan."""
        try:
            if not isinstance(plan, dict):
                return False
            
            total_weight = 0
            for symbol, alloc in plan.items():
                if not isinstance(alloc, dict):
                    return False
                if 'weight' not in alloc or 'reason' not in alloc:
                    return False
                if not isinstance(alloc['weight'], (int, float)):
                    return False
                total_weight += alloc['weight']
            
            # Allow small floating point error
            return abs(total_weight - 1.0) < 0.01
            
        except Exception as e:
            logger.error(f"Plan validation error: {str(e)}")
            return False

    def _generate_fallback_plan(self, features):
        """Generate a simple fallback plan if GPT fails."""
        plan = {}
        n_tickers = len(features)
        if n_tickers > 0:
            weight = 1.0 / n_tickers
            for symbol in features:
                plan[symbol] = {
                    'weight': weight,
                    'reason': 'Fallback: Equal weight allocation'
                }
        else:
            plan['cash'] = {
                'weight': 1.0,
                'reason': 'Fallback: 100% cash'
            }
        return self.validate_plan(plan)
    
    def justify_plan(self, plan, context):
        """
        Provide justification for the proposed plan.
        """
        if not plan:
            return "No plan to justify."
        lines = []
        for symbol, alloc in plan.items():
            reason = alloc.get('reason', 'No reason provided')
            weight = alloc.get('weight', 0)
            lines.append(f"{symbol}: weight {weight:.2f}, reason: {reason}")
        return "; ".join(lines)

    def critique_plan(self, plan, context):
        """
        Critique another agent's plan from a momentum perspective.
        """
        if not plan:
            return "No plan to critique."
        critiques = []
        for symbol, alloc in plan.items():
            weight = alloc.get('weight', 0)
            reason = alloc.get('reason', '')
            if weight > 0.5:
                critiques.append(f"{symbol}: High allocation ({weight:.2f})—momentum strategies typically diversify more unless a very strong trend is present.")
            if 'momentum' not in reason.lower():
                critiques.append(f"{symbol}: Reason does not mention momentum—consider trend strength.")
        if not critiques:
            return "Plan aligns with momentum principles."
        return "; ".join(critiques)

    def validate_constraints(self, plan, constraints):
        """
        Validate the plan against explicit, implicit, and derived constraints.
        Returns (is_valid, violations)
        """
        violations = []
        if not plan:
            violations.append("Plan is empty.")
            return False, violations
        total_weight = sum(alloc.get('weight', 0) for alloc in plan.values())
        if abs(total_weight - 1.0) > 0.01:
            violations.append(f"Total weight {total_weight:.2f} does not sum to 1.0.")
        for symbol, alloc in plan.items():
            if alloc.get('weight', 0) < 0:
                violations.append(f"{symbol}: Negative weight.")
            if alloc.get('weight', 0) > 1:
                violations.append(f"{symbol}: Weight exceeds 1.")
        is_valid = len(violations) == 0
        return is_valid, violations

    def execute(self, plan):
        """
        Simulate execution of the plan (mock logic).
        """
        if not plan:
            return {'status': 'failed', 'reason': 'No plan to execute.'}
        # Simulate success
        return {'status': 'success', 'plan': plan} 