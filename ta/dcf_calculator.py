import numpy as np
from typing import List, Dict, Tuple

class DCFCalculator:
    """
    Discounted Cash Flow (DCF) Model Calculator
    
    This calculator performs enterprise valuation using the DCF method,
    projecting free cash flows and discounting them to present value.
    """
    
    def __init__(self, 
                 initial_fcf: float,
                 projection_years: int = 5,
                 growth_rates: List[float] = None,
                 terminal_growth_rate: float = 0.025,
                 wacc: float = 0.10,
                 cash: float = 0,
                 debt: float = 0,
                 shares_outstanding: float = None):
        """
        Initialize DCF Calculator
        
        Args:
            initial_fcf: Most recent Free Cash Flow (in millions)
            projection_years: Number of years to project (default: 5)
            growth_rates: List of growth rates for each projection year
            terminal_growth_rate: Perpetual growth rate (default: 2.5%)
            wacc: Weighted Average Cost of Capital (default: 10%)
            cash: Cash and cash equivalents (in millions)
            debt: Total debt (in millions)
            shares_outstanding: Number of shares outstanding (in millions)
        """
        self.initial_fcf = initial_fcf
        self.projection_years = projection_years
        self.terminal_growth_rate = terminal_growth_rate
        self.wacc = wacc
        self.cash = cash
        self.debt = debt
        self.shares_outstanding = shares_outstanding
        
        # Set growth rates
        if growth_rates is None:
            self.growth_rates = [0.10] * projection_years
        else:
            if len(growth_rates) != projection_years:
                raise ValueError(f"Growth rates list must have {projection_years} elements")
            self.growth_rates = growth_rates
    
    def project_fcf(self) -> List[float]:
        """Project Free Cash Flows for the forecast period"""
        fcf_projections = []
        current_fcf = self.initial_fcf
        
        for growth_rate in self.growth_rates:
            current_fcf = current_fcf * (1 + growth_rate)
            fcf_projections.append(current_fcf)
        
        return fcf_projections
    
    def calculate_terminal_value(self, final_fcf: float) -> float:
        """
        Calculate Terminal Value using Gordon Growth Model
        
        TV = FCF(n+1) / (WACC - g)
        where FCF(n+1) = FCF(n) * (1 + terminal_growth_rate)
        """
        fcf_terminal_year = final_fcf * (1 + self.terminal_growth_rate)
        terminal_value = fcf_terminal_year / (self.wacc - self.terminal_growth_rate)
        return terminal_value
    
    def discount_cash_flows(self, cash_flows: List[float]) -> Tuple[List[float], float]:
        """
        Discount projected cash flows to present value
        
        Returns:
            Tuple of (discounted cash flows list, sum of discounted cash flows)
        """
        discounted_cfs = []
        
        for year, cf in enumerate(cash_flows, start=1):
            pv = cf / ((1 + self.wacc) ** year)
            discounted_cfs.append(pv)
        
        return discounted_cfs, sum(discounted_cfs)
    
    def calculate_enterprise_value(self) -> Dict[str, float]:
        """Calculate Enterprise Value and Equity Value"""
        # Project FCFs
        fcf_projections = self.project_fcf()
        
        # Calculate Terminal Value
        terminal_value = self.calculate_terminal_value(fcf_projections[-1])
        
        # Discount FCFs
        discounted_fcfs, pv_fcf = self.discount_cash_flows(fcf_projections)
        
        # Discount Terminal Value
        pv_terminal_value = terminal_value / ((1 + self.wacc) ** self.projection_years)
        
        # Enterprise Value
        enterprise_value = pv_fcf + pv_terminal_value
        
        # Equity Value
        equity_value = enterprise_value + self.cash - self.debt
        
        # Price per share
        price_per_share = equity_value / self.shares_outstanding if self.shares_outstanding else None
        
        return {
            'fcf_projections': fcf_projections,
            'discounted_fcfs': discounted_fcfs,
            'pv_fcf': pv_fcf,
            'terminal_value': terminal_value,
            'pv_terminal_value': pv_terminal_value,
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'price_per_share': price_per_share
        }
    
    def sensitivity_analysis(self, 
                           wacc_range: List[float] = None,
                           terminal_growth_range: List[float] = None) -> np.ndarray:
        """
        Perform sensitivity analysis on WACC and Terminal Growth Rate
        
        Args:
            wacc_range: List of WACC values to test
            terminal_growth_range: List of terminal growth rates to test
            
        Returns:
            2D numpy array of equity values
        """
        if wacc_range is None:
            wacc_range = [self.wacc - 0.02, self.wacc - 0.01, self.wacc, 
                         self.wacc + 0.01, self.wacc + 0.02]
        
        if terminal_growth_range is None:
            terminal_growth_range = [self.terminal_growth_rate - 0.01, 
                                    self.terminal_growth_rate - 0.005,
                                    self.terminal_growth_rate,
                                    self.terminal_growth_rate + 0.005,
                                    self.terminal_growth_rate + 0.01]
        
        results = np.zeros((len(wacc_range), len(terminal_growth_range)))
        
        original_wacc = self.wacc
        original_tg = self.terminal_growth_rate
        
        for i, wacc in enumerate(wacc_range):
            for j, tg in enumerate(terminal_growth_range):
                self.wacc = wacc
                self.terminal_growth_rate = tg
                valuation = self.calculate_enterprise_value()
                results[i, j] = valuation['equity_value']
        
        # Restore original values
        self.wacc = original_wacc
        self.terminal_growth_rate = original_tg
        
        return results, wacc_range, terminal_growth_range
    
    def print_summary(self):
        """Print a formatted summary of the DCF valuation"""
        results = self.calculate_enterprise_value()
        
        print("=" * 60)
        print("DCF VALUATION SUMMARY")
        print("=" * 60)
        print(f"\nInitial FCF: ${self.initial_fcf:,.2f}M")
        print(f"WACC: {self.wacc*100:.2f}%")
        print(f"Terminal Growth Rate: {self.terminal_growth_rate*100:.2f}%")
        
        print(f"\n{'Year':<8} {'Growth Rate':<15} {'FCF':<15} {'PV of FCF':<15}")
        print("-" * 60)
        
        for i in range(self.projection_years):
            print(f"{i+1:<8} {self.growth_rates[i]*100:>12.2f}%  "
                  f"${results['fcf_projections'][i]:>12,.2f}M  "
                  f"${results['discounted_fcfs'][i]:>12,.2f}M")
        
        print("\n" + "=" * 60)
        print(f"PV of Projected FCFs:        ${results['pv_fcf']:>15,.2f}M")
        print(f"Terminal Value:              ${results['terminal_value']:>15,.2f}M")
        print(f"PV of Terminal Value:        ${results['pv_terminal_value']:>15,.2f}M")
        print("-" * 60)
        print(f"Enterprise Value:            ${results['enterprise_value']:>15,.2f}M")
        print(f"Plus: Cash:                  ${self.cash:>15,.2f}M")
        print(f"Less: Debt:                  ${self.debt:>15,.2f}M")
        print("-" * 60)
        print(f"Equity Value:                ${results['equity_value']:>15,.2f}M")
        
        if results['price_per_share']:
            print(f"\nShares Outstanding:          {self.shares_outstanding:>15,.2f}M")
            print(f"Price per Share:             ${results['price_per_share']:>15,.2f}")
        
        print("=" * 60)


def get_float_input(prompt: str, default: float = None) -> float:
    """Helper function to get float input from user"""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if user_input == "":
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_int_input(prompt: str, default: int = None) -> int:
    """Helper function to get integer input from user"""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if user_input == "":
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            return int(user_input)
        except ValueError:
            print("Invalid input. Please enter a whole number.")


def get_yes_no(prompt: str) -> bool:
    """Helper function to get yes/no input"""
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")


def run_interactive_dcf():
    """Run interactive DCF calculator"""
    print("=" * 60)
    print("DCF MODEL CALCULATOR")
    print("=" * 60)
    print("All values in millions\n")

    # Get basic inputs
    initial_fcf = get_float_input("Initial FCF")

    projection_years = get_int_input("Projection period (years)", default=5)

    # Get growth rates
    print(f"\nGrowth rates (decimal format):")
    growth_rates = []
    for i in range(projection_years):
        rate = get_float_input(f"  Year {i+1}")
        growth_rates.append(rate)

    # Get terminal growth rate
    terminal_growth_rate = get_float_input("\nTerminal growth rate", default=0.025)

    # Get WACC
    wacc = get_float_input("WACC", default=0.10)

    # Get balance sheet items
    print()
    cash = get_float_input("Cash & equivalents", default=0)
    debt = get_float_input("Total debt", default=0)

    # Get shares outstanding
    has_shares = get_yes_no("\nCalculate price per share?")
    shares_outstanding = None
    if has_shares:
        shares_outstanding = get_float_input("Shares outstanding")

    # Create DCF calculator
    print("\n" + "=" * 60)
    print("CALCULATING VALUATION...")
    print("=" * 60 + "\n")

    dcf = DCFCalculator(
        initial_fcf=initial_fcf,
        projection_years=projection_years,
        growth_rates=growth_rates,
        terminal_growth_rate=terminal_growth_rate,
        wacc=wacc,
        cash=cash,
        debt=debt,
        shares_outstanding=shares_outstanding
    )

    # Print results
    dcf.print_summary()

    # Ask if user wants sensitivity analysis
    if get_yes_no("\nWould you like to see a sensitivity analysis?"):
        print("\n\nSENSITIVITY ANALYSIS")
        print("=" * 60)
        results, wacc_range, tg_range = dcf.sensitivity_analysis()

        print(f"\nEquity Value Sensitivity (in millions)")
        header = "WACC \\ Term.Growth"
        print(f"{header:<15}", end="")
        for tg in tg_range:
            print(f"{tg*100:>12.2f}%", end="")
        print()
        print("-" * 75)
        
        for i, wacc_val in enumerate(wacc_range):
            print(f"{wacc_val*100:>12.2f}%     ", end="")
            for j in range(len(tg_range)):
                print(f"${results[i,j]:>11,.0f}M", end="")
            print()
        print()


# Run the interactive calculator
if __name__ == "__main__":
    run_interactive_dcf()
