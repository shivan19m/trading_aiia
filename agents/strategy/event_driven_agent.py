import os
from dotenv import load_dotenv
load_dotenv()  # Ensure .env is loaded for API key
import openai
import json
from agents.base import BaseAgent
import random

class EventDrivenAgent(BaseAgent):
    """
    Agent that generates trade plans based on earnings, news sentiment, and macroeconomic events using GPT-4.
    """
    def __init__(self):
        # Model used: gpt-4
        super().__init__()
    # def propose_plan(self, features, context, memory_agent=None):
    #     """
    #     Propose a portfolio allocation plan using event-driven indicators, vector memory retrieval, and GPT-4.
    #     features: dict of {symbol: feature_dict}
    #     context: string (optional extra context)
    #     memory_agent: MemoryAgent instance (optional)
    #     Returns: dict {symbol: {weight, reason}, ...}
    #     """
    #     import openai
    #     import os
    #     import json
    #     openai.api_key = os.getenv('OPENAI_API_KEY')
    #     # 1. Construct context_str from key indicators
    #     context_str = "; ".join([
    #         f"{symbol} RSI {vals.get('rsi', 'nan'):.2f}, MACD {vals.get('macd', 'nan'):.2f}, VolRatio {vals.get('avg_vol_ratio', 'nan'):.2f}"
    #         for symbol, vals in features.items()
    #     ])
    #     # 2. Retrieve similar plans
    #     similar_plans = []
    #     if memory_agent:
    #         similar_plans = memory_agent.retrieve_similar_plans(context_str, top_k=3)
    #     similar_str = "\n".join([str(plan) for plan in similar_plans])
    #     # 3. Prepare prompt
    #     features_str = "\n".join([f"{symbol}: {vals}" for symbol, vals in features.items()])
    #     prompt = (
    #         "You are an event-driven portfolio manager. Given the following features for multiple tickers, "
    #         "and these similar past plans:\n"
    #         f"{similar_str}\n"
    #         "generate a portfolio allocation plan. Use event-driven indicators (news sentiment, earnings, macro events, volume spikes) "
    #         "to decide which assets to overweight or underweight. "
    #         "Return ONLY a valid JSON object mapping each symbol to a dict with keys: weight (0-1), reason (string). "
    #         "Include a 'cash' key if you want to hold cash. Do not include any explanation, markdown, or code block formatting.\n"
    #         f"Features:\n{features_str}"
    #     )
    #     try:
    #         response = openai.chat.completions.create(
    #             model="gpt-4",
    #             messages=[{"role": "system", "content": "You are a financial trading assistant."},
    #                       {"role": "user", "content": prompt}],
    #             temperature=0.3,
    #             max_tokens=500
    #         )
    #         plan_str = response.choices[0].message.content.strip()
    #         try:
    #             plan = json.loads(plan_str)
    #         except Exception as je:
    #             print(f"[OpenAI JSON Parse Error] {je}")
    #             print(f"[OpenAI API Output] {plan_str}")
    #             raise je
    #     except Exception as e:
    #         print(f"[OpenAI API Error] {e}")
    #         n = len(features)
    #         alloc_weight = 0.8 / n if n > 0 else 0
    #         plan = {symbol: {"weight": alloc_weight, "reason": "Event-driven fallback allocation."} for symbol in features.keys()}
    #         plan["cash"] = {"weight": 0.2, "reason": "Hold cash."}
    #     # 4. Store plan in vector memory
    #     if memory_agent:
    #         memory_agent.store_plan_vector(plan, context_str)
    #     return plan

    def propose_plan(self, features, context, memory_agent=None):
        """
        Propose a portfolio allocation plan using event-driven indicators, vector memory retrieval, and GPT API.
        """
        import openai
        import os
        import json


        # ✅ 1. Construct context_str safely
        def format_feature_line(symbol, vals):
            if isinstance(vals, dict):
                try:
                    return (
                        f"{symbol} RSI {float(vals.get('rsi', 'nan')):.2f}, "
                        f"MACD {float(vals.get('macd', 'nan')):.2f}, "
                        f"VolRatio {float(vals.get('avg_vol_ratio', 'nan')):.2f}"
                    )
                except Exception:
                    return f"{symbol} has malformed numeric values."
            else:
                return f"{symbol} has invalid feature data."

        context_str = "; ".join([format_feature_line(sym, val) for sym, val in features.items()])

        # ✅ 2. Retrieve similar plans
        similar_plans = []
        if memory_agent:
            similar_plans = memory_agent.retrieve_similar_plans(context_str, top_k=3)
        similar_str = "\n".join([str(plan) for plan in similar_plans])

        # ✅ 3. Prepare prompt (cleaned)
        features_str = "\n".join([f"{sym}: {val}" for sym, val in features.items()])
        prompt = (
            "You are an event-driven portfolio manager. Given the following features for multiple tickers, "
            "and these similar past plans:\n"
            f"{similar_str}\n"
            "Generate a portfolio allocation plan. Use event-driven indicators (news sentiment, earnings, macro events, volume spikes) "
            "to decide which assets to overweight or underweight. "
            "Return ONLY a valid JSON object mapping each symbol to a dict with keys: weight (0-1), reason (string). "
            "Include a 'cash' key if you want to hold cash. Do not include any explanation, markdown, or code block formatting.\n"
            f"Features:\n{features_str}"
        )

        # ✅ 4. GPT API call with fallback
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # ✅ replace with a model you have access to
                messages=[
                    {"role": "system", "content": "You are a financial trading assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            plan_str = response.choices[0].message.content.strip()
            try:
                plan = json.loads(plan_str)
            except Exception as je:
                print(f"[OpenAI JSON Parse Error] {je}")
                print(f"[OpenAI API Output] {plan_str}")
                raise je

        except Exception as e:
            print(f"[OpenAI API Error] {e}")
            n = len(features)
            alloc_weight = 0.8 / n if n > 0 else 0
            plan = {
                sym: {"weight": alloc_weight, "reason": "Event-driven fallback allocation."}
                for sym in features.keys()
            }
            plan["cash"] = {"weight": 0.2, "reason": "Hold cash."}

        # ✅ 5. Store plan to memory
        if memory_agent:
            memory_agent.store_plan_vector(plan, context_str)

        return plan

    # def justify_plan(self, plan, context):
    #     """
    #     Use GPT-4 to justify the event-driven plan.
    #     """
    #     prompt = (
    #         f"Justify the following event-driven trade plan using news sentiment, macro events, and historical reactions: {plan}. "
    #         "Return a concise explanation."
    #     )
    #     try:
    #         response = openai.chat.completions.create(
    #             model="gpt-4",
    #             messages=[{"role": "system", "content": "You are a financial trading assistant."},
    #                       {"role": "user", "content": prompt}],
    #             temperature=0.3,
    #             max_tokens=200
    #         )
    #         justification = response.choices[0].message.content
    #         return justification
    #     except Exception as e:
    #         print(f"[OpenAI API Error] {e}")
    #         return f"Plan justified by recent news sentiment and macro event analysis. Action: {plan['action']} {plan['quantity']} shares of {plan['symbol']}."

    def justify_plan(self, plan, context):
        """
        Use GPT to justify the event-driven portfolio allocation plan.
        """
        import openai
        import os

        # Format the plan as text for GPT input
        plan_str = "\n".join([
            f"{symbol}: weight {info.get('weight', 0):.2f}, reason: {info.get('reason', 'N/A')}"
            for symbol, info in plan.items()
        ])

        prompt = (
            "You are a financial assistant. Justify the following event-driven portfolio plan using news sentiment, "
            "macro events, and historical price reactions.\n"
            f"{plan_str}\n"
            "Give a concise summary of the allocation rationale."
        )

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # ✅ 使用你有权限的模型
                messages=[
                    {"role": "system", "content": "You are a financial trading assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[OpenAI API Error] {e}")
            # Fallback justification — format for portfolio instead of per-trade
            fallback = "\n".join([
                f"{symbol}: Allocate {info.get('weight', 0):.2f}, reason: {info.get('reason', 'Fallback allocation')}."
                for symbol, info in plan.items()
            ])
            return "Fallback justification based on event-driven heuristics:\n" + fallback

    def critique_plan(self, plan, context):
        """
        TODO: Implement Socratic critique logic using LLM.
        """
        import openai
        import os
        system_prompt = "You are an event-driven trading assistant. Critique trading plans for their alignment with recent earnings, news sentiment, and macroeconomic events."
        user_prompt = (
            f"Critique the following plan from an event-driven perspective. Does it make sense based on recent earnings, news sentiment, and macroeconomic events?\n"
            f"Plan: {plan}"
        )
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            critique = response.choices[0].message.content.strip()
            return critique
        except Exception as e:
            return "EventDrivenAgent critique: Could not generate critique due to API error. Check if the plan considers recent earnings, news sentiment, and macro events."

    def validate_constraints(self, plan, constraints):
        """
        Validate the plan against constraints (mock logic).
        """
        return True

    def execute(self, plan):
        """
        Simulate execution of the plan.
        """
        return {'status': 'executed', 'plan': plan} 