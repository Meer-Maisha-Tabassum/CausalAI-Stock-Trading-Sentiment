from dowhy import CausalModel, causal_estimators
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import graphviz

class causal_model:
    def __init__(self, data, treatment, outcome, common_causes, instruments):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes
        self.instruments = instruments
        self.model = None
        self.identified_estimand = None
        self.estimate_regression = None
        self.estimate_iv = None
        self.refute_results_regression = None
        self.refute_results_iv = None

    def create_model(self):
        self.model = CausalModel(
            data=self.data,
            treatment=self.treatment,
            outcome=self.outcome,
            common_causes=self.common_causes,
            instruments=self.instruments
        )

    def view_model(self, save_path='static/causal_model.png'):
        # self.model.view_model()
        # plt.savefig(save_path)
        # plt.close()
        # return save_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Manually create the graph using graphviz
        dot = graphviz.Digraph(format='png')

        # Add nodes for treatment and outcome
        dot.node('T', self.treatment, shape='ellipse', color='red')
        dot.node('Y', self.outcome, shape='ellipse', color='blue')
        
        # Add nodes for common causes
        for cc in self.common_causes:
            dot.node(cc, cc, shape='ellipse', color='green')
        
        # Add nodes for instruments
        for instr in self.instruments:
            dot.node(instr, instr, shape='ellipse', color='purple')

        # Add edges for treatment and outcome
        dot.edge('T', 'Y', label='Treatment')

        # Add edges for common causes
        for cc in self.common_causes:
            dot.edge(cc, 'T', label='Common Cause')
            dot.edge(cc, 'Y', label='Common Cause')

        # Add edges for instruments
        for instr in self.instruments:
            dot.edge(instr, 'T', label='Instrument')

        # Save the graph
        save_path_without_ext = os.path.splitext(save_path)[0]
        dot.render(save_path_without_ext, cleanup=True)

        return save_path


    def identify_effect(self):
        self.identified_estimand = self.model.identify_effect()

    def estimate_effects(self):
        self.estimate_regression = self.model.estimate_effect(
            self.identified_estimand, method_name="backdoor.linear_regression"
        )
        self.estimate_iv = self.model.estimate_effect(
            self.identified_estimand, method_name="iv.instrumental_variable"
        )

    def refute_estimates(self):
        # Print the summary of data used for estimation
        print("Summary of data used for estimation:")
        print(self.data.describe())
        
        # Check for NaNs and infinite values
        print("Checking for NaNs and infinite values:")
        print("NaNs per column:\n", self.data.isna().sum())
        print("Infinite values per column:\n", np.isinf(self.data).sum())
        
        self.refute_results_regression = self.model.refute_estimate(
            self.identified_estimand, self.estimate_regression, method_name="random_common_cause"
        )
        self.refute_results_iv = self.model.refute_estimate(
            self.identified_estimand, self.estimate_iv, method_name="placebo_treatment_refuter", placebo_type="permute"
        )

    def print_results(self):
        print("Causal effect estimate (linear regression):", self.estimate_regression.value)
        print("Causal effect estimate (instrumental variable):", self.estimate_iv.value)
        print("Refute results (linear regression):", self.refute_results_regression.estimated_effect)
        print("Refute results (instrumental variable):", self.refute_results_iv)
        
    def get_results(self):
        results = {
            "Causal effect estimate (linear regression)": self.estimate_regression,
            "Causal effect estimate (instrumental variable)": self.estimate_iv,
            "Refute results (linear regression)": self.refute_results_regression,
            "Refute results (instrumental variable)": self.refute_results_iv
        }
        
        return results
    
    def get_estimate_regression(self):
        return self.estimate_regression.value
    
    def get_estimate_iv(self):
        return self.estimate_iv.value
    
    def get_pos_neg_lr(self):
        value = self.get_estimate_regression()
        if value < 0:
            return "NEGETIVE"
        elif value > 0:
            return "POSITIVE"
        else:
            return "NEUTRAL"
    
    def get_pos_neg_iv(self):
        value = self.get_estimate_iv()
        if value < 0:
            return "NEGETIVE"
        elif value > 0:
            return "POSITIVE"
        else:
            return "NEUTRAL"

    def get_increase_decrease_lr(self):
        value = self.get_estimate_regression()
        if value < 0:
            return "DECREASE"
        elif value > 0:
            return "INCREASE"
        else:
            return "NO EFFECT"
        
    def get_increase_decrease_iv(self):
        value = self.get_estimate_iv()
        if value < 0:
            return "DECREASE"
        elif value > 0:
            return "INCREASE"
        else:
            return "NO EFFECT"

# import pandas as pd
# import numpy as np
# from dowhy import CausalModel
# import matplotlib.pyplot as plt
# import os

# # Create a demo DataFrame with simulated stock data
# np.random.seed(42)
# data = pd.DataFrame({
#     'Close': np.random.normal(100, 10, 100),  # Simulated stock closing prices
#     'RSI': np.random.uniform(0, 100, 100),    # Simulated RSI values
#     'Volume': np.random.normal(1000, 100, 100),  # Simulated trading volume
#     'MACD': np.random.normal(0, 5, 100)  # Simulated MACD values
# })

# # Binarize the 'RSI' variable for the purpose of the causal analysis
# data['RSI_binary'] = pd.qcut(data['RSI'], q=2, labels=[0, 1]).astype(int)

# class CausalModelAnalysis:
#     def __init__(self, data, treatment, outcome, common_causes, instruments):
#         self.data = data
#         self.treatment = treatment
#         self.outcome = outcome
#         self.common_causes = common_causes
#         self.instruments = instruments
#         self.model = None
#         self.identified_estimand = None
#         self.estimate_regression = None
#         self.estimate_iv = None
#         self.refute_results_regression = None
#         self.refute_results_iv = None
#         self.refute_results_unobserved_confounder = None

#     def create_model(self):
#         self.model = CausalModel(
#             data=self.data,
#             treatment=self.treatment,
#             outcome=self.outcome,
#             common_causes=self.common_causes,
#             instruments=self.instruments
#         )

#     def view_model(self, save_path='static/causal_model.png'):
#         self.model.view_model()
#         plt.savefig(save_path)
#         plt.close()
#         return save_path

#     def identify_effect(self):
#         self.identified_estimand = self.model.identify_effect()

#     def estimate_effects(self):
#         self.estimate_regression = self.model.estimate_effect(
#             self.identified_estimand, method_name="backdoor.linear_regression"
#         )
#         self.estimate_iv = self.model.estimate_effect(
#             self.identified_estimand, method_name="iv.instrumental_variable"
#         )

#     def refute_estimates(self):
#         self.refute_results_regression = self.model.refute_estimate(
#             self.identified_estimand, self.estimate_regression, method_name="random_common_cause"
#         )
#         self.refute_results_iv = self.model.refute_estimate(
#             self.identified_estimand, self.estimate_iv, method_name="placebo_treatment_refuter", placebo_type="permute"
#         )

#     def refute_unobserved_confounder(self):
#         self.refute_results_unobserved_confounder = self.model.refute_estimate(
#             self.identified_estimand, self.estimate_regression, method_name="add_unobserved_common_cause"
#         )

#     def get_regression_estimate(self):
#         return self.estimate_regression

#     def get_iv_estimate(self):
#         return self.estimate_iv

#     def get_regression_refutation(self):
#         return self.refute_results_regression

#     def get_iv_refutation(self):
#         return self.refute_results_iv

#     def get_unobserved_confounder_refutation(self):
#         return self.refute_results_unobserved_confounder

#     def print_results(self):
#         print("\nCausal Effect Estimate (Linear Regression):")
#         print(self.get_regression_estimate())

#         print("\nCausal Effect Estimate (Instrumental Variable):")
#         print(self.get_iv_estimate())

#         print("\nRefutation Results (Linear Regression):")
#         print(self.get_regression_refutation())

#         print("\nRefutation Results (Instrumental Variable):")
#         print(self.get_iv_refutation())

#         print("\nRefutation Results (Unobserved Confounder):")
#         print(self.get_unobserved_confounder_refutation())

# # Example usage
# if __name__ == "__main__":
#     treatment = 'RSI_binary'  # Use the binarized RSI variable
#     outcome = 'Close'
#     common_causes = ['MACD']
#     instruments = ['Volume']  # Add instruments if you have any

#     # Create an instance of the CausalModelAnalysis class
#     causal_model_instance = CausalModelAnalysis(data, treatment, outcome, common_causes, instruments)

#     # Create the causal model
#     print("Creating causal model...")
#     causal_model_instance.create_model()

#     # View the model (graphically)
#     print("Viewing causal model...")
#     model_image_path = causal_model_instance.view_model()
#     print(f"Causal model graph saved at: {model_image_path}")

#     # Identify the causal effect
#     print("Identifying causal effect...")
#     causal_model_instance.identify_effect()

#     # Estimate the effects
#     print("Estimating causal effects...")
#     causal_model_instance.estimate_effects()

#     # Refute the estimates
#     print("Refuting the estimates...")
#     causal_model_instance.refute_estimates()

#     # Refute with unobserved confounder
#     print("Refuting the estimates with unobserved confounder...")
#     causal_model_instance.refute_unobserved_confounder()

#     # Print the results in a structured manner
#     print("Printing the results...")
#     causal_model_instance.print_results()



# import logging
# import numpy as np
# from dowhy import CausalModel
# import matplotlib.pyplot as plt
# import json

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, filename='causal_model.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# def convert_to_serializable(obj):
#     if isinstance(obj, np.generic):
#         return obj.item()
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         return obj

# class causal_model:
#     def __init__(self, data, treatment, outcome, common_causes, instruments):
#         self.data = data
#         self.treatment = treatment
#         self.outcome = outcome
#         self.common_causes = common_causes
#         self.instruments = instruments
#         self.model = None
#         self.identified_estimand = None
#         self.estimate_regression = None
#         self.estimate_iv = None
#         self.refute_results_regression = None
#         self.refute_results_iv = None

#     def create_model(self):
#         self.model = CausalModel(
#             data=self.data,
#             treatment=self.treatment,
#             outcome=self.outcome,
#             common_causes=self.common_causes,
#             instruments=self.instruments
#         )

#     def view_model(self, save_path='static/causal_model.png'):
#         self.model.view_model()
#         plt.savefig(save_path)
#         plt.close()
#         return save_path

#     def identify_effect(self):
#         self.identified_estimand = self.model.identify_effect()

#     def estimate_effects(self):
#         self.estimate_regression = self.model.estimate_effect(
#             self.identified_estimand, method_name="backdoor.linear_regression"
#         )
#         self.estimate_iv = self.model.estimate_effect(
#             self.identified_estimand, method_name="iv.instrumental_variable"
#         )

#     def refute_estimates(self):
#         self.refute_results_regression = self.model.refute_estimate(
#             self.identified_estimand, self.estimate_regression, method_name="random_common_cause"
#         )
#         self.refute_results_iv = self.model.refute_estimate(
#             self.identified_estimand, self.estimate_iv, method_name="placebo_treatment_refuter", placebo_type="permute"
#         )

#     def get_results(self):
#         results = {
#             "Causal effect estimate (linear regression)": self._serialize_estimate(self.estimate_regression),
#             "Causal effect estimate (instrumental variable)": self._serialize_estimate(self.estimate_iv),
#             "Refute results (linear regression)": self._serialize_refute(self.refute_results_regression),
#             "Refute results (instrumental variable)": self._serialize_refute(self.refute_results_iv)
#         }
#         return results

#     def _serialize_estimate(self, estimate):
#         if estimate is None:
#             return None

#         try:
#             estimand = estimate.target_estimand.estimands
#             logging.debug(f"Estimands structure: {estimand}")
#             estimand_key = next(iter(estimand), None)
#             if not estimand_key:
#                 return None
            
#             estimand_data = estimand.get(estimand_key, {})
#             assumptions = estimand_data.get("assumptions", {})
#             assumptions_list = [assumptions.get(key, "N/A") for key in assumptions]

#             return {
#                 "Identified estimand": {
#                     "Estimand type": str(estimate.target_estimand.estimand_type),
#                     "Estimand name": estimand_key,
#                     "Estimand expression": str(estimand_data.get("estimand", "N/A")),
#                     "Estimand assumptions": assumptions_list
#                 },
#                 "Realized estimand": convert_to_serializable(getattr(estimate, 'realized_estimand', 'N/A')),
#                 "Estimate": {
#                     "Mean value": convert_to_serializable(estimate.value),
#                     "Effect strength": convert_to_serializable(getattr(estimate, 'effect_strength', None)),
#                     "p value": convert_to_serializable(getattr(estimate, 'p_value', None))
#                 }
#             }
#         except Exception as e:
#             logging.error(f"Error serializing estimate: {e}")
#             return None

#     def _serialize_refute(self, refute):
#         if refute is None:
#             return None
#         return {
#             "Refutation result": refute.refutation_result,
#             "Method name": getattr(refute, 'method_name', 'N/A'),
#             "Estimated effect": convert_to_serializable(refute.estimated_effect),
#             "New effect": convert_to_serializable(refute.new_effect),
#             "p value": convert_to_serializable(getattr(refute, 'p_value', 'N/A'))
#         }




# from dowhy import CausalModel
# import matplotlib.pyplot as plt
# import os

# class causal_model:
#     def __init__(self, data, treatment, outcome, common_causes, instruments, graph):
#         self.data = data
#         self.treatment = treatment
#         self.outcome = outcome
#         self.common_causes = common_causes
#         self.instruments = instruments
#         self.graph = graph
#         self.model = None
#         self.identified_estimand = None
#         self.estimate_regression = None
#         self.estimate_iv = None
#         self.refute_results_regression = None
#         self.refute_results_iv = None

#     def create_model(self):
#         self.model = CausalModel(
#             data=self.data,
#             treatment=self.treatment,
#             outcome=self.outcome,
#             common_causes=self.common_causes,
#             instruments=self.instruments,
#             graph=self.graph
#         )

#     def view_model(self, save_path='static/causal_model.png'):
#         self.model.view_model()
#         plt.savefig(save_path)
#         plt.close()
#         return save_path

#     def identify_effect(self):
#         self.identified_estimand = self.model.identify_effect()

#     def estimate_effects(self):
#         try:
#             self.estimate_regression = self.model.estimate_effect(
#                 self.identified_estimand, method_name="backdoor.linear_regression"
#             )
#         except Exception as e:
#             print(f"Error estimating effect using linear regression: {e}")
#             self.estimate_regression = None

#         try:
#             self.estimate_iv = self.model.estimate_effect(
#                 self.identified_estimand, method_name="iv.instrumental_variable"
#             )
#         except Exception as e:
#             print(f"Error estimating effect using instrumental variable: {e}")
#             self.estimate_iv = None

#     def refute_estimates(self):
#         if self.estimate_regression is not None:
#             try:
#                 self.refute_results_regression = self.model.refute_estimate(
#                     self.identified_estimand, self.estimate_regression, method_name="random_common_cause"
#                 )
#             except Exception as e:
#                 print(f"Error refuting estimate (linear regression): {e}")
#         else:
#             print("No valid regression estimate to refute.")

#         if self.estimate_iv is not None:
#             try:
#                 self.refute_results_iv = self.model.refute_estimate(
#                     self.identified_estimand, self.estimate_iv, method_name="placebo_treatment_refuter", placebo_type="permute"
#                 )
#             except Exception as e:
#                 print(f"Error refuting estimate (instrumental variable): {e}")
#         else:
#             print("No valid IV estimate to refute.")

#     def print_results(self):
#         print("Causal effect estimate (linear regression):", self.estimate_regression)
#         print("Causal effect estimate (instrumental variable):", self.estimate_iv)
#         print("Refute results (linear regression):", self.refute_results_regression)
#         print("Refute results (instrumental variable):", self.refute_results_iv)
        
#     def get_results(self):
#         results = {
#             "Causal effect estimate (linear regression)": self.estimate_regression,
#             "Causal effect estimate (instrumental variable)": self.estimate_iv,
#             "Refute results (linear regression)": self.refute_results_regression,
#             "Refute results (instrumental variable)": self.refute_results_iv
#         }
#         return results