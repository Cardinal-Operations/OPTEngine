import argparse
from modules.tsp_generator_v2 import TSPGenerator
from modules.knapsack_generator_v2 import KnapsackGenerator
from modules.binpacking_generator import BinPackingGenerator
from modules.jobshop_generator import JobShopGenerator
from modules.netflow_generator import NetFlowGenerator
from modules.inventory_generator import Inventory_generator
from modules.pollution_generator import PollutionGenerator
from modules.portfolio_generator_v2 import Portfolio_generator
from modules.transportation_generator import TransportationGenerator
from modules.production_generator import ProductionGenerator

def _size_overrides(problem_type: str, problem_size: str) -> dict:
    """
    Return kwargs that override the 'size-driving' ranges based on problem_type and problem_size.
    Only keys that affect complexity are overridden; everything else remains as you set above.
    """
    if problem_size not in {"easy", "medium", "hard", "all"}:
        return {}

    # Convenience helpers
    E, M, H, A= "easy", "medium", "hard", "all"

    # Per-problem size maps: override ONLY the main size dimensions.
    maps = {
        "tsp": {
            E: {"n_cities_range": (4, 10)},
            M: {"n_cities_range": (10, 30)},
            H: {"n_cities_range": (30, 50)},
            A: {"n_cities_range": (4, 50)},
        },
        "knapsack": {
            E: {"n_items_range": (5, 10)},
            M: {"n_items_range": (10, 20)},
            H: {"n_items_range": (20, 30)},
            A: {"n_items_range": (5, 30)},
        },
        "binpacking": {
            E: {"n_items_range": (10, 18)},
            M: {"n_items_range": (18, 30)},
            H: {"n_items_range": (30, 41)},
            A: {"n_items_range": (10, 41)},
        },
        "jobshop": {
            E: {"job_range": (3, 5)},
            M: {"job_range": (5, 8)},
            H: {"job_range": (8, 10)},
            A: {"job_range": (5, 10)},
        },
        "netflow": {
            E: {"n_nodes_range": (5, 6)},
            M: {"n_nodes_range": (6, 8)},
            H: {"n_nodes_range": (8, 10)},
            A: {"n_nodes_range": (5, 10)},
        },
        "inventory": { #done
            E: {"T_range": (5, 16)},
            M: {"T_range": (16, 26)},
            H: {"T_range": (26, 41)},
            A: {"T_range": (5, 41)},
        },
        "pollution": {
            # Scale both time periods (sources) and number of methods
            E: {"T_range": (3, 5), "K_range": (2, 3)},
            M: {"T_range": (5, 8), "K_range": (3, 4)},
            H: {"T_range": (8, 11), "K_range": (4, 6)},
            A: {"T_range": (3, 11), "K_range": (2, 6)},
        },
        "portfolio": { # done
            E: {"I_range": (5, 16)},
            M: {"I_range": (16, 26)},
            H: {"I_range": (26, 36)},
            A: {"I_range": (5, 36)},
        },
        "transportation": { # done
            # Scale both supply (n) and demand (m) counts
            E: {"n_range": (3, 7), "m_range": (3, 7)},
            M: {"n_range": (7, 10), "m_range": (7, 10)},
            H: {"n_range": (10, 15), "m_range": (10, 15)},
            A: {"T_range": (3, 15), "K_range": (3, 15)},
        },
        "production": { # done
            # Scale products (I) and resources (J) 
            E: {"I_range": (2, 5), "J_range": (2, 6)},
            M: {"I_range": (5, 8), "J_range": (6, 9)},
            H: {"I_range": (8, 12), "J_range": (9, 13)},
            A: {"I_range": (2, 12), "J_range": (2, 13)},
        },
    }

    return maps.get(problem_type, {}).get(problem_size, {})


def generate_instances(problem_type: str, problem_size: str | None = None):
    # Compute overrides once
    overrides = _size_overrides(problem_type, problem_size) if problem_size else {}

    if problem_type == "tsp":
        kwargs = dict(
            n_cities_range=(4, 40),
            coord_range=(0, 200),
            samples_per_type=20,
            seed=0,
        )
        kwargs.update(overrides)
        generator = TSPGenerator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "knapsack":
        kwargs = dict(
            n_items_range=(5, 30),
            weight_range=(1, 50),
            value_range=(10, 300),
            capacity_ratio=0.7,
            samples_per_type=10,
            seed=0,
        )
        kwargs.update(overrides)
        generator = KnapsackGenerator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "binpacking":
        kwargs = dict(
            n_items_range=(10, 41),
            bin_capacity=100,
            weight_range=(30, 70),
            samples_per_type=10,
            seed=0,
        )
        kwargs.update(overrides)
        generator = BinPackingGenerator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "jobshop":
        kwargs = dict(
            job_range=(3, 10),
            samples_per_type=10,
            time_range=(1, 10),
            seed=0,
        )
        kwargs.update(overrides)
        generator = JobShopGenerator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "netflow":
        kwargs = dict(
            n_nodes_range=(5, 10),
            supply_range=(10, 100),
            demand_range=(10, 100),
            shipping_cost_range=(1, 10),
            capacity_range=(5, 100),
            samples_per_type=10,
            seed=0,
        )
        kwargs.update(overrides)
        generator = NetFlowGenerator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "inventory":
        kwargs = dict(
            T_range=(5, 50),
            demand_range=(10, 60),
            I0_range=(0, 100),
            Qmin_range=(0, 20),
            Qmax_range=(20, 80),
            lead_range=(0, 4),
            p_range=(1, 6),
            h_range=(1, 3),
            c_range=(6, 15),
            capacity_factor=(0.8, 1.6),
            samples_per_T=10,
            seed=0,
        )
        kwargs.update(overrides)
        generator = Inventory_generator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "pollution":
        kwargs = dict(
            T_range=(3, 11),
            K_range=(2, 6),
            w_range=(0.5, 3.0),
            p_range=(50.0, 300.0),
            s_range=(0.10, 0.90),
            P_range=(0.20, 0.70),
            cost_range=(10.0, 200.0),
            samples_per_size=10,
            seed=42,
        )
        kwargs.update(overrides)
        generator = PollutionGenerator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "portfolio":
        kwargs = dict(
            I_range=(5, 21),
            r_range=(0.05, 0.20),
            v_range=(0.01, 0.30),
            l_max=0.10,
            u_minmax=(0.30, 0.90),
            # Lmin_range=(0.20, 0.60),
            Lmin_range=(0.60, 0.80),
            Rmin_factor=(0.60, 0.95),
            # Vmax_factor=(1.00, 1.50),
            Vmax_factor=(0.6, 1.0),
            samples_per_I=10,
            seed=42,
        )
        kwargs.update(overrides)
        generator = Portfolio_generator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "transportation":
        kwargs = dict(
            n_range=(3, 11),
            m_range=(3, 11),
            supply_range=(50, 200),
            demand_share=(0.60, 0.95),
            cost_range=(1, 20),
            samples_per_size=10,
            seed=42,
        )
        kwargs.update(overrides)
        generator = TransportationGenerator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "production":
        kwargs = dict(
            I_range=(3, 11),
            J_range=(2, 6),
            profit_range=(5.0, 20.0),
            time_range=(0.2, 2.0),
            ref_x_total_range=(50.0, 200.0),
            capacity_relax=(1.00, 1.50),
            samples_per_size=5,
            seed=0,
        )
        kwargs.update(overrides)
        generator = ProductionGenerator(**kwargs)
        generator.generate_instances()
        generator.map_to_nl()

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


def main():
    parser = argparse.ArgumentParser(description="Generate problem instances and NL mapping.")
    parser.add_argument(
        "--problem_type",
        type=str,
        required=True,
        help=("Type of problem to generate "
              "(tsp, knapsack, binpacking, jobshop, netflow, inventory, pollution, "
              "portfolio, transportation, production)")
    )
    parser.add_argument(
        "--problem_size",
        type=str,
        choices=["easy", "medium", "hard", "all"],
        help="Optional complexity level; if omitted, defaults to the original ranges."
    )
    args = parser.parse_args()
    generate_instances(args.problem_type, args.problem_size)


if __name__ == "__main__":
    main()