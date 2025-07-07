import sys


def verbose_print(msg, aisles, cutoff=25):
    print(f"    {msg} {len(aisles)} ({sorted(list(aisles))[:cutoff]}{'...' if len(aisles) > cutoff else ''})")


def progress_bar(msg, progress, total, length=50):
    percent = int(100 * (progress / float(total)))
    filled = int(length * progress // total)
    bar = '█' * filled + '-' * (length - filled)
    sys.stdout.write(f'{msg} |{bar}| {percent}%\n')
    sys.stdout.flush()


# Función para calcular capacidad máxima posible
def calculate_max_capacity(orders, aisles, aisle_indices):
    total_capacity_per_item = {}
    current_aisles_data = [aisles[i] for i in aisle_indices]
    for aisle in current_aisles_data:
        for item_id, capacity in aisle['items'].items():
            total_capacity_per_item[item_id] = total_capacity_per_item.get(item_id, 0) + capacity

    max_possible_quantity = 0
    for order in orders:
        can_satisfy_order = True
        for item_id, demand in order['items'].items():
            available_capacity = total_capacity_per_item.get(item_id, 0)
            if available_capacity < demand:
                can_satisfy_order = False
                break
        if can_satisfy_order:
            max_possible_quantity += order['total_quantity']
    return max_possible_quantity
