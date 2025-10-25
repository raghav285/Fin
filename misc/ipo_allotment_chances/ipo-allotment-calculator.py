import math
from typing import List, Tuple, Dict
from collections import Counter

def calculate_combinations(n: int, r: int) -> float:
    return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

def calculate_probability(n: int, x: int, p: float) -> float:
    """Calculate probability of getting at least x successes in n trials"""
    total_prob = 0
    for i in range(x, n + 1):
        prob = calculate_combinations(n, i) * (p ** i) * ((1 - p) ** (n - i))
        total_prob += prob
    return total_prob

def parse_application_input(input_str: str) -> List[str]:
    """Parse input string into list of categories"""
    categories = []
    parts = input_str.lower().split()
    
    i = 0
    while i < len(parts):
        if parts[i].isdigit() and i + 1 < len(parts):
            count = int(parts[i])
            category = parts[i + 1]
            if category not in ['retail', 'shni', 'bhni']:
                raise ValueError(f"Invalid category: {category}")
            categories.extend([category] * count)
            i += 2
        else:
            if parts[i] not in ['retail', 'shni', 'bhni']:
                raise ValueError(f"Invalid category: {parts[i]}")
            categories.append(parts[i])
            i += 1
            
    return categories

def get_subscription_ratio(category: str) -> float:
    while True:
        try:
            subscription = float(input(f"Enter subscription ratio for {category.upper()}: "))
            if subscription <= 0:
                print("Subscription ratio must be positive!")
                continue
            return subscription
        except ValueError:
            print("Please enter a valid number!")

def get_application_details() -> List[str]:
    print("\n=== IPO Allotment Chance Calculator ===")
    print("\nAvailable categories:")
    print("1. retail - Retail Individual Investor")
    print("2. shni  - Small HNI")
    print("3. bhni  - Big HNI")
    print("\nEnter applications (e.g., '2 retail bhni 3 shni'):")
    
    while True:
        try:
            input_str = input("> ")
            categories = parse_application_input(input_str)
            if not categories:
                print("Please enter at least one category!")
                continue
            return categories
        except ValueError as e:
            print(f"Error: {e}")
            print("Please try again using format like '2 retail bhni 3 shni'")

def main():
    categories = get_application_details()
    print("\nYou applied from these categories:", " + ".join(categories))
    
    # Get subscription details for each unique category
    unique_categories = set(categories)
    subscription_ratios = {}
    
    print("\nEnter subscription ratios:")
    for category in unique_categories:
        subscription_ratios[category] = get_subscription_ratio(category)
    
    # Process each category type separately
    category_counts = Counter(categories)
    
    print("\nProbability Calculations:")
    print("-" * 40)
    
    for category, count in category_counts.items():
        effective_subscription = subscription_ratios[category]
        if category == 'bhni':
            effective_subscription = subscription_ratios[category] / 5
            print(f"\n{category.upper()} (Adjusted subscription ratio: {effective_subscription:.2f}x)")
        else:
            print(f"\n{category.upper()} (Subscription ratio: {effective_subscription:.2f}x)")
        
        p = 1 / effective_subscription
        
        for i in range(1, count + 1):
            prob = calculate_probability(count, i, p) * 100
            print(f"Probability of getting at least {i} lot(s): {prob:.2f}%")

if __name__ == "__main__":
    try:
        main()
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
