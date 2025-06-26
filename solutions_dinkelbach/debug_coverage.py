import cplex
import sys
import logging
from dinkelbach_solver import read_instance

def debug_coverage_forcing():
    """Debug why all aisles are being forced"""
    
    # Read instance
    orders, aisles, wave_size_lb, wave_size_ub = read_instance('instance_0003.txt')
    
    # Get all items
    items = set()
    for order in orders:
        items.update(order.keys())
    
    print(f"Total items in instance: {len(items)}")
    print(f"Total aisles: {len(aisles)}")
    
    # Check how many aisles are needed for each item
    items_forcing_aisles = 0
    for k in items:
        # Find aisles that contain this item
        aisles_with_item = []
        for j, aisle in enumerate(aisles):
            if k in aisle:
                aisles_with_item.append(j)
        
        # Find orders that need this item
        orders_with_item = []
        total_needed = 0
        for i, order in enumerate(orders):
            if k in order:
                orders_with_item.append(i)
                total_needed += order[k]
        
        if len(aisles_with_item) == 1:
            items_forcing_aisles += 1
            print(f"Item {k}: FORCES aisle {aisles_with_item[0]} (only option)")
        elif len(aisles_with_item) == 0:
            print(f"Item {k}: NO AISLES AVAILABLE! (infeasible)")
    
    print(f"\nItems that force specific aisles: {items_forcing_aisles}")
    print(f"This could explain why many aisles are used")
    
    # Check if there are items that appear in many orders
    item_frequency = {}
    for order in orders:
        for item_id in order.keys():
            if item_id not in item_frequency:
                item_frequency[item_id] = 0
            item_frequency[item_id] += 1
    
    frequent_items = [(item_id, freq) for item_id, freq in item_frequency.items() if freq > 5]
    print(f"\nItems appearing in >5 orders: {len(frequent_items)}")
    for item_id, freq in sorted(frequent_items, key=lambda x: x[1], reverse=True)[:10]:
        aisles_with_item = [j for j, aisle in enumerate(aisles) if item_id in aisle]
        print(f"  Item {item_id}: {freq} orders, {len(aisles_with_item)} aisles")

if __name__ == "__main__":
    debug_coverage_forcing() 