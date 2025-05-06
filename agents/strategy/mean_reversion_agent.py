import os
from dotenv import load_dotenv
load_dotenv()  # Ensure .env is loaded for API key
import openai
import json
from agents.base import BaseAgent
import random

class MeanReversionAgent(BaseAgent):
    """
    Agent that generates trade plans based on statistical deviations from equilibrium using GPT-4.
    """
    def __init__(self):
        # Model used: gpt-4o-mini
        super().__init__()

    def propose_plan(self, features, context, memory_agent=None):
        """
        Propose a portfolio allocation plan using mean-reversion indicators, vector memory retrieval, and GPT-4.
        features: dict of {symbol: feature_dict}
        context: string (optional extra context)
        memory_agent: MemoryAgent instance (optional)
        Returns: dict {symbol: {weight, reason}, ...}
        """
        # 1. Construct context_str from key indicators
        # context_str = "; ".join([
        #     f"{symbol} z-score {vals.get('zscore', 'nan'):.2f}, BB {vals.get('bb_ma', 'nan'):.2f}, RSI {vals.get('rsi', 'nan'):.2f}"
        #     for symbol, vals in features.items()
        # ])
        context_str = "; ".join([
            (
                f"{symbol} z-score {vals.get('zscore', float('nan')):.2f}, "
                f"BB {vals.get('bb_ma', float('nan')):.2f}, "
                f"RSI {vals.get('rsi', float('nan')):.2f}"
                if isinstance(vals, dict)
                else f"{symbol} has invalid feature data: {vals}"
            )
            for symbol, vals in features.items()
        ])
        # 2. Retrieve similar plans
        similar_plans = []
        if memory_agent:
            similar_plans = memory_agent.retrieve_similar_plans(context_str, top_k=3)
        similar_str = "\n".join([str(plan) for plan in similar_plans])
        # 3. Prepare prompt
        features_str = "\n".join([f"{symbol}: {vals}" for symbol, vals in features.items()])
        prompt = (
            "You are a mean-reversion-based portfolio manager. Given the following features for multiple tickers, "
            "and these similar past plans:\n"
            f"{similar_str}\n"
            "generate a portfolio allocation plan. Use mean-reversion indicators (z-score, Bollinger Bands, RSI) "
            "to decide which assets to overweight or underweight. "
            "Return ONLY a valid JSON object mapping each symbol to a dict with keys: weight (0-1), reason (string). "
            "Include a 'cash' key if you want to hold cash. Do not include any explanation, markdown, or code block formatting.\n"
            f"Features:\n{features_str}"
        )
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a financial trading assistant."},
                          {"role": "user", "content": prompt}],
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
            plan = {symbol: {"weight": alloc_weight, "reason": "Mean reversion fallback allocation."} for symbol in features.keys()}
            plan["cash"] = {"weight": 0.2, "reason": "Hold cash."}
        # 4. Store plan in vector memory
        if memory_agent:
            memory_agent.store_plan_vector(plan, context_str)
        return plan

    # def justify_plan(self, plan, context):
    #     """
    #     Use GPT-4 to justify the mean-reversion-based plan.
    #     """
    #     prompt = (
    #         f"Justify the following mean-reversion trade plan using z-scores, Bollinger Bands, or mean-reversion models: {plan}. "
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
    #         return f"Plan justified by deviation from historical mean (z-score/Bollinger Band). Action: {plan['action']} {plan['quantity']} shares of {plan['symbol']}."

    def justify_plan(self, plan, context):
        """
        Use GPT-4 to justify the mean-reversion-based multi-asset plan.
        """

        plan_summary = "\n".join([
            f"{symbol}: weight {vals.get('weight', 0):.2f}, reason: {vals.get('reason', 'N/A')}"
            for symbol, vals in plan.items()
        ])

        prompt = (
            "You are a financial trading assistant. Given the following mean-reversion-based portfolio allocation plan:\n"
            f"{plan_summary}\n"
            "Explain the rationale behind this plan based on z-scores, Bollinger Bands, RSI, and general mean-reversion principles. "
            "Be concise and focus on the logic behind overweighting or underweighting each asset."
        )

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Replace with "gpt-3.5-turbo" if you lack GPT-4 access
                messages=[{"role": "system", "content": "You are a financial trading assistant."},
                        {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI API Error] {e}")
            # Fallback explanation
            fallback_lines = [
                f"{symbol}: Allocate {vals.get('weight', 0):.2f} based on mean-reversion signal (e.g., z-score, BB, RSI)."
                for symbol, vals in plan.items()
            ]
            return "Fallback justification:\n" + "\n".join(fallback_lines)

    def critique_plan(self, plan, context):
        """
        Use GPT-4 to critique the plan based on mean-reversion logic, z-scores, and Bollinger Bands.
        """
        system_prompt = "You are a mean-reversion-based trading assistant. Critique trading plans for their alignment with mean reversion logic, z-scores, and Bollinger Bands."
        user_prompt = (
            f"Critique the following plan from a mean reversion perspective. Does it make sense based on statistical deviations from the mean, z-scores, and Bollinger Bands?\n"
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
            return "MeanReversionAgent critique: Could not generate critique due to API error. Check if the plan is supported by mean reversion indicators, z-scores, and Bollinger Bands."

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