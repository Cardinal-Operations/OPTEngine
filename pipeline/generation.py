import argparse
from modules.tsp_generator import TSPGenerator
from modules.knapsack_generator_obj import KnapsackGenerator
from modules.binpacking_generator_data import BinPackingGenerator
from modules.jobshop_generator_obj import JobShopGenerator
from modules.netflow_generator_data import NetFlowGenerator
from modules.inventory_generator_obj import Inventory_generator
from modules.pollution_generator_data import PollutionGenerator
from modules.portfolio_generator_data import Portfolio_generator
from modules.transportation_generator_data import TransportationGenerator
from modules.production_generator_data import ProductionGenerator

def generate_instances(problem_type: str):
    if problem_type == "tsp":
        generator = TSPGenerator(
            n_cities_range=(4, 20),
            coord_range=(0, 200),
            samples_per_type=10,
            seed=0
        )
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "knapsack":
        generator = KnapsackGenerator(
            n_items_range=(5, 30),
            weight_range=(1, 50),
            value_range=(10, 300),
            capacity_ratio=0.7,
            samples_per_type=10,
            seed=0
        )
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "binpacking":
        generator = BinPackingGenerator(
            n_items_range=(10, 41),
            bin_capacity=100,
            weight_range=(30, 70),
            samples_per_type=10,
            seed=0
        )
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "jobshop":
        generator = JobShopGenerator(
            job_range=(3, 10),
            samples_per_type=10,
            time_range=(1, 10),
            seed=0
        )
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "netflow":
        generator = NetFlowGenerator(
            n_nodes_range=(3, 15),
            supply_range=(10, 100),
            demand_range=(10, 100),
            shipping_cost_range=(1, 10),
            capacity_range=(5, 100),
            samples_per_type=10,
            seed=0
        )
        generator.generate_instances()
        generator.map_to_nl()
    elif problem_type == "inventory":
        generator = Inventory_generator(
            T_range=(5, 21),
            demand_range=(10, 60),
            I0_range=(0, 100),
            Qmin_range=(0, 20),
            Qmax_range=(20, 80),
            # lead_range=(0, 4),
            p_range=(1, 6),
            h_range=(1, 3),
            c_range=(6, 15),
            capacity_factor=(0.8, 1.6),
            samples_per_T=10,
            seed=0
        )
        generator.generate_instances()
        generator.map_to_nl()
    elif problem_type == "pollution":
        generator = PollutionGenerator(
            T_range=(3, 11),
            K_range=(2, 6),
            w_range=(0.5, 3.0), 
            p_range=(50.0, 300.0), 
            s_range=(0.10, 0.90),
            P_range=(0.20, 0.70),
            cost_range=(10.0, 200.0), 
            samples_per_size=10, 
            seed=42
        )
        
        generator.generate_instances()
        generator.map_to_nl()
    elif problem_type == "portfolio":
        generator = Portfolio_generator(
            I_range=(5, 21),
            r_range=(0.02, 0.20),
            v_range=(0.01, 0.30),
            l_max=0.10,
            u_minmax=(0.30, 0.90),
            Lmin_range=(0.20, 0.60),
            Rmin_factor=(0.60, 0.95),
            Vmax_factor=(1.00, 1.50),
            samples_per_I=10,
            seed=42
        )
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "transportation":
        generator = TransportationGenerator(
            n_range=(3, 11),
            m_range=(3, 11),
            supply_range=(50, 200),
            demand_share=(0.60, 0.95),
            cost_range=(1, 20),
            samples_per_size=5,
            seed=42
        )
        generator.generate_instances()
        generator.map_to_nl()

    elif problem_type == "production":
        generator = ProductionGenerator(
            I_range=(3, 11),         
            J_range=(2, 6),
            profit_range=(5.0, 20.0),
            time_range=(0.2, 2.0),
            ref_x_total_range=(50.0, 200.0),
            capacity_relax=(1.00, 1.50),
            samples_per_size=5,
            #fixed_cost_range=(100.0, 150.0),
            seed=0
        )
        generator.generate_instances()
        generator.map_to_nl()

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


def main():
    parser = argparse.ArgumentParser(description="Generate problem instances and NL mapping.")
    parser.add_argument("--problem_type", type=str, required=True,
                        help="Type of problem to generate (tsp, knapsack, binpacking, jobshop, netflow, inventory, pollution, portfolio, transportation, production)")
    args = parser.parse_args()
    generate_instances(args.problem_type)


if __name__ == "__main__":
    main()