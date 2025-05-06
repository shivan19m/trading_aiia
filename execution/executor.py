from agents.base import BaseAgent
import math
import logging

logger = logging.getLogger(__name__)

class ExecutorAgent(BaseAgent):

    def __init__(self, initial_cash=100000.0, slippage=0.001, commission=1.0):

        super().__init__()
        self.cash = initial_cash
        self.holdings = {}  
        self.trade_log = []
        self.slippage = slippage
        self.commission = commission
        self.tickers = set()  

    def set_tickers(self, tickers):
        
        self.tickers = set(tickers or [])

    def execute(self, plan):
        
        instructions = self._normalize_plan(plan)

        sell_instr = [i for i in instructions if i["action"].upper() == "SELL"]
        buy_instr = [i for i in instructions if i["action"].upper() == "BUY"]
        hold_instr = [i for i in instructions if i["action"].upper() == "HOLD"]
        ordered = sell_instr + buy_instr + hold_instr

        trade_results = []

        portfolio_value = self._calc_portfolio_value(ordered)

        for instr in ordered:
            tkr = instr["ticker"]
            action = instr["action"].upper()
            weight = instr.get("weight", 0.0)
            date = instr.get("date", None)  
            price = instr.get("price", None)

            log_entry = {
                "date": date,
                "ticker": tkr,
                "action": action,
                "weight": weight,
                "price": None,
                "quantity": 0.0,
                "commission": 0.0,
                "slippage": self.slippage,
                "cost": 0.0,    
                "status": "NO_ACTION",
                "reason": "Not executed"
            }

            if self.tickers and tkr not in self.tickers:
                log_entry["status"] = "ROLLED_BACK"
                log_entry["reason"] = f"Unknown or disallowed ticker: {tkr}"
                trade_results.append(log_entry)
                self.trade_log.append(log_entry)
                continue

            if tkr not in self.holdings:
                self.holdings[tkr] = 0.0

            if not price:
         
                log_entry["status"] = "ROLLED_BACK"
                log_entry["reason"] = f"No price provided for {tkr}"
                trade_results.append(log_entry)
                self.trade_log.append(log_entry)
                continue

            if action == "BUY":
                effective_price = price * (1 + self.slippage)
            elif action == "SELL":
                effective_price = price * (1 - self.slippage)
            else:
                effective_price = price

            log_entry["price"] = effective_price

            quantity = 0.0
            current_value = self.holdings[tkr] * price

            if action == "BUY":
                desired_value = portfolio_value * weight
                delta_value = desired_value - current_value
                if delta_value <= 0:

                    log_entry["reason"] = "Allocation already met"
                    trade_results.append(log_entry)
                    self.trade_log.append(log_entry)
                    continue
                quantity = delta_value / effective_price

            elif action == "SELL":
                if weight <= 0.0:

                    quantity = self.holdings[tkr]
                else:
                    desired_value = portfolio_value * weight
                    delta_value = current_value - desired_value
                    if delta_value <= 0:

                        log_entry["reason"] = "Allocation not exceeded"
                        trade_results.append(log_entry)
                        self.trade_log.append(log_entry)
                        continue
                    quantity = delta_value / effective_price

                if quantity > self.holdings[tkr]:
                    quantity = self.holdings[tkr]

            elif action == "HOLD":

                log_entry["reason"] = "HOLD action => no trade"
                trade_results.append(log_entry)
                self.trade_log.append(log_entry)
                continue
            else:
                log_entry["status"] = "ROLLED_BACK"
                log_entry["reason"] = f"Unknown action: {action}"
                trade_results.append(log_entry)
                self.trade_log.append(log_entry)
                continue

            if quantity <= 0:
                log_entry["reason"] = "Quantity <= 0, no trade"
                trade_results.append(log_entry)
                self.trade_log.append(log_entry)
                continue

            quantity = round(quantity, 6)

            if action == "BUY":
                cost = quantity * effective_price + self.commission
                if cost > self.cash:

                    log_entry["status"] = "ROLLED_BACK"
                    log_entry["reason"] = "Insufficient cash"
                else:

                    self.cash -= cost
                    self.holdings[tkr] += quantity
                    log_entry["status"] = "EXECUTED"
                    log_entry["reason"] = "Buy completed"
                    log_entry["quantity"] = quantity
                    log_entry["commission"] = self.commission
                    log_entry["cost"] = cost

            elif action == "SELL":

                if quantity > self.holdings[tkr]:
                    log_entry["status"] = "ROLLED_BACK"
                    log_entry["reason"] = "Not enough shares to sell"
                else:
                    proceeds = quantity * effective_price - self.commission
                    if proceeds < 0:

                        log_entry["status"] = "ROLLED_BACK"
                        log_entry["reason"] = "Negative proceeds"
                    else:
                        self.holdings[tkr] -= quantity
                        self.cash += proceeds
                        log_entry["status"] = "EXECUTED"
                        log_entry["reason"] = "Sell completed"
                        log_entry["quantity"] = quantity
                        log_entry["commission"] = self.commission
                        log_entry["cost"] = proceeds

            trade_results.append(log_entry)
            self.trade_log.append(log_entry)

        rolled_back = any(t["status"] == "ROLLED_BACK" for t in trade_results)
        if rolled_back:
            overall_status = "completed_with_errors"
        else:
            overall_status = "success"

        report = {
            "status": overall_status,
            "cash_after": self.cash,
            "holdings_after": dict(self.holdings),
            "trades": trade_results,
        }
        return report

    def _normalize_plan(self, plan):
        """Convert either dict-based or list-based plan into a uniform list of instructions."""
        instructions = []
        if isinstance(plan, dict):
            for tkr, info in plan.items():
                action = info.get("action", "HOLD")
                weight = info.get("weight", 0.0)

                instr = {
                    "ticker": tkr,
                    "action": action,
                    "weight": weight,
                    "price": info.get("price", None),
                    "date": info.get("date", None)
                }
                instructions.append(instr)
        elif isinstance(plan, list):

            instructions = plan
        else:
            raise ValueError("Plan must be a dict or a list of instructions.")
        return instructions

    def _calc_portfolio_value(self, instructions):

        total_value = self.cash
        for tkr, shares in self.holdings.items():
            if shares <= 0:
                continue

            relevant = [x for x in instructions if x["ticker"] == tkr and x.get("price") is not None]

            if relevant:
                price = relevant[0]["price"]
                total_value += shares * price
            else:

                pass
        return total_value

    def propose_plan(self, market_data, context):
        pass

    def justify_plan(self, plan, context):
        pass

    def critique_plan(self, plan, context):
        pass

    def validate_constraints(self, plan, constraints):
        pass
