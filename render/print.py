class RenderPrint:
    """Vertical farm logger made to render OpenAI Gym environment to the console"""

    def __init__(self, start_datetime):
        self.start_datetime = start_datetime

        self.action_info = {
            0: "u_rot",
            1: "u_fan",
            2: "u_cool",
            3: "u_heat",
            4: "u_humid",
            5: "u_c_inj",
            6: "u_light"
        }

        self.data_info = {
            0: "T_out",
            1: "RH_out",
            2: "Electricity price",
            3: "Light input"
        }

    def render(self, terminated, current_step, t, model, solutions, climate_attrs, crop_attrs, actions, all_data, index):
        print("Start date:", self.start_datetime)
        print("=========== STEP", current_step, "============")
        print("--- Solution ---")
        model.print_attributes()
        print()

        if climate_attrs:
            print("--- Climate attributes ---")
            for key in climate_attrs.keys():
                print(f"{key}: {climate_attrs[key][index]}")
            print()

        if crop_attrs:
            print("--- Crop attributes ---")
            for key in crop_attrs.keys():
                print(f"{key}: {crop_attrs[key][index]}")
            print()

        print("--- Control inputs ---")
        for i, action in enumerate(actions[:,index]):
            print(f"{self.action_info[i]}: {action}")

        print()
        print("--- Data ---")
        for i, data in enumerate(all_data[:,index]):
            print(f"{self.data_info[i]}: {data}")
        print("================================")
        print()

    def close(self):
        pass

