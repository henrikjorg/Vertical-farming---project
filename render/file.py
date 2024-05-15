import csv
import datetime
import os

class RenderFile:
    """Vertical farm render made to save OpenAI Gym environment to csv file"""
    def __init__(self, config, start_date, t, y):
        self.start_date = start_date

        now_datetime_str = datetime.datetime.now().strftime("%d%m%y-%H%M")
        self.file_name = '../render/csv/' + now_datetime_str + '_simulation.csv'

        # Create file if it doesn't exist
        dir_name = os.path.dirname(self.file_name)
        os.makedirs(dir_name, exist_ok=True)

        self.action_info = {
            0: "u_rot",
            1: "u_fan",
            2: "u_cool",
            3: "u_heat",
            4: "u_humid",
            5: "u_c_inj"
        }

        self.y_labels = ["T_in", "Chi_in", "CO2_in", "T_env", "T_sup", "Chi_sup", "X_ns", "X_s"]

        self.action_labels = ["u_rot", "u_fan", "u_cool", "u_heat", "u_humid", "u_c_inj"]

        self.data_info = {
            0: "T_out",
            1: "RH_out",
            2: "Electricity price",
            3: "u_light"
        }

        self.csv_file = None

    def render(self, terminated, current_step, t, model, solutions, climate_attrs, crop_attrs, actions, all_data, index):
        pass

    def save(self, t, solutions, climate_attrs, crop_attrs, actions, all_data):
        self.csv_file = open(self.file_name, 'w', newline='')
        writer = csv.writer(self.csv_file)

        # Create the header row
        header = ["Date", "t"] + self.y_labels + list(climate_attrs.keys()) + list(crop_attrs.keys()) + self.action_labels + list(self.data_info.values())
        writer.writerow(header)

        data = []

        for i in range(len(t)):
            data_row = [self.start_date + datetime.timedelta(seconds=t[i]), t[i]]

            for j in range(solutions.shape[0]):
                data_row.append(solutions[j, i])
            # Add climate and crop data
            if climate_attrs:
                for value in climate_attrs.values():
                    data_row.append(value[i])
            if crop_attrs:
                for value in crop_attrs.values():
                    data_row.append(value[i])

            # Add actions
            for action in actions[:, i]:
                data_row.append(action)

            # Add data
            for d in all_data[:, i]:
                data_row.append(d)

            data.append(data_row)

        # Write the data row
        writer = csv.writer(self.csv_file)
        writer.writerows(data)

    def close(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None