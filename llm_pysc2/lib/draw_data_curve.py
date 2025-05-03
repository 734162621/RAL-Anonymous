import json
from matplotlib import pyplot as plt
import scienceplots
import numpy as np
import os

data_path = ".\\sc2_rl_agent\\starcraftenv_test\\log\\chatgpt_log\\D3\\"
data_path_baseline = ".\\sc2_rl_agent\\starcraftenv_test\\log\\chatgpt_log\\原方法D3\\"

files = []
dirs = [data_path]

while len(dirs) > 0:
    curdir = dirs.pop()
    for f in os.listdir(curdir):
        if os.path.isdir(os.path.join(curdir, f)):
            dirs.append(os.path.join(curdir, f))
        else:
            files.append(os.path.join(curdir, f))
files = [f for f in files if f.endswith("raw_observation.json")]

files_baseline = []
dirs_baseline = [data_path_baseline]

while len(dirs_baseline) > 0:
    curdir = dirs_baseline.pop()
    for f in os.listdir(curdir):
        if os.path.isdir(os.path.join(curdir, f)):
            dirs_baseline.append(os.path.join(curdir, f))
        else:
            files_baseline.append(os.path.join(curdir, f))
files_baseline = [f for f in files_baseline if f.endswith("raw_observation.json")]

all_game_time = []
all_worker_supply = []
all_mineral = []
all_gas = []
all_supply_left = []
all_supply_cap = []
all_supply_used = []
all_army_supply = []
for f in files:
    with open(f, "r") as file:
        file.readline()
        game_time_list = []
        worker_supply_list = []
        mineral_list = []
        gas_list = []
        supply_left_list = []
        supply_cap_list = []
        supply_used_list = []
        army_supply_list = []
        last_game_time = 0
        for line in file:
            data = json.loads(line)
            resource = data["resource"]
            resource = resource.replace("'", '"')
            resource = json.loads(resource)
            game_time = resource["game_time"]  # "'00:00'"
            game_time = game_time.replace("'", "")
            game_time = game_time.split(":")
            game_time = int(game_time[0]) * 60 + int(game_time[1])
            if game_time <= last_game_time:
                continue
            last_game_time = game_time
            game_time_list.append(game_time)
            worker_supply_list.append(resource["worker_supply"])
            mineral_list.append(resource["mineral"])
            gas_list.append(resource["gas"])
            supply_left_list.append(resource["supply_left"])
            supply_cap_list.append(resource["supply_cap"])
            supply_used_list.append(resource["supply_used"])
            army_supply_list.append(resource["army_supply"])

        all_game_time.append(game_time_list)
        all_worker_supply.append(worker_supply_list)
        all_mineral.append(mineral_list)
        all_gas.append(gas_list)
        all_supply_left.append(supply_left_list)
        all_supply_cap.append(supply_cap_list)
        all_supply_used.append(supply_used_list)
        all_army_supply.append(army_supply_list)
        
max_game_time = max([len(game_time) for game_time in all_game_time])

all_game_time_baseline = []
all_worker_supply_baseline = []
all_mineral_baseline = []
all_gas_baseline = []
all_supply_left_baseline = []
all_supply_cap_baseline = []
all_supply_used_baseline = []
all_army_supply_baseline = []

for f in files_baseline:
    with open(f, "r") as file:
        file.readline()
        game_time_list = []
        worker_supply_list = []
        mineral_list = []
        gas_list = []
        supply_left_list = []
        supply_cap_list = []
        supply_used_list = []
        army_supply_list = []
        last_game_time = 0
        for line in file:
            data = json.loads(line)
            resource = data["resource"]
            resource = resource.replace("'", '"')
            resource = json.loads(resource)
            game_time = resource["game_time"]  # "'00:00'"
            game_time = game_time.replace("'", "")
            game_time = game_time.split(":")
            game_time = int(game_time[0]) * 60 + int(game_time[1])
            if game_time <= last_game_time:
                continue
            last_game_time = game_time
            game_time_list.append(game_time)
            worker_supply_list.append(resource["worker_supply"])
            mineral_list.append(resource["mineral"])
            gas_list.append(resource["gas"])
            supply_left_list.append(resource["supply_left"])
            supply_cap_list.append(resource["supply_cap"])
            supply_used_list.append(resource["supply_used"])
            army_supply_list.append(resource["army_supply"])

        all_game_time_baseline.append(game_time_list)
        all_worker_supply_baseline.append(worker_supply_list)
        all_mineral_baseline.append(mineral_list)
        all_gas_baseline.append(gas_list)
        all_supply_left_baseline.append(supply_left_list)
        all_supply_cap_baseline.append(supply_cap_list)
        all_supply_used_baseline.append(supply_used_list)
        all_army_supply_baseline.append(army_supply_list)
        
max_game_time_baseline = max([len(game_time) for game_time in all_game_time_baseline])

all_accumulated_mineral = []
all_accumulated_gas = []
all_worker_work_time = []

for i in range(len(all_game_time)):
    accumulated_mineral = [0]
    for j in range(1, len(all_game_time[i])):
        if all_mineral[i][j] > all_mineral[i][j - 1]:
            accumulated_mineral.append(
                accumulated_mineral[-1] + all_mineral[i][j] - all_mineral[i][j - 1]
            )
    all_accumulated_mineral.append(accumulated_mineral)
    accumulated_gas = [0]
    for j in range(1, len(all_game_time[i])):
        if all_gas[i][j] > all_gas[i][j - 1]:
            accumulated_gas.append(
                accumulated_gas[-1] + all_gas[i][j] - all_gas[i][j - 1]
            )
    all_accumulated_gas.append(accumulated_gas)
    worker_work_time = [0]
    for j in range(1, len(all_game_time[i])):
        worker_work_time.append(worker_work_time[-1] + all_worker_supply[i][j])
    all_worker_work_time.append(worker_work_time)

all_accumulated_mineral_baseline = []
all_accumulated_gas_baseline = []
all_worker_work_time_baseline = []

for i in range(len(all_game_time_baseline)):
    accumulated_mineral_baseline = [0]
    for j in range(1, len(all_game_time_baseline[i])):
        if all_mineral_baseline[i][j] > all_mineral_baseline[i][j - 1]:
            accumulated_mineral_baseline.append(
                accumulated_mineral_baseline[-1]
                + all_mineral_baseline[i][j]
                - all_mineral_baseline[i][j - 1]
            )
    all_accumulated_mineral_baseline.append(accumulated_mineral_baseline)
    accumulated_gas_baseline = [0]
    for j in range(1, len(all_game_time_baseline[i])):
        if all_gas_baseline[i][j] > all_gas_baseline[i][j - 1]:
            accumulated_gas_baseline.append(
                accumulated_gas_baseline[-1]
                + all_gas_baseline[i][j]
                - all_gas_baseline[i][j - 1]
            )
    all_accumulated_gas_baseline.append(accumulated_gas_baseline)
    worker_work_time_baseline = [0]
    for j in range(1, len(all_game_time_baseline[i])):
        worker_work_time_baseline.append(
            worker_work_time_baseline[-1] + all_worker_supply_baseline[i][j]
        )
    all_worker_work_time_baseline.append(worker_work_time_baseline)

mean_accumulated_mineral_list = []
std_accumulated_mineral_list_upper = []
std_accumulated_mineral_list_lower = []
mean_accumulated_gas_list = []
std_accumulated_gas_list_upper = []
std_accumulated_gas_list_lower = []
mean_worker_work_time_list = []
std_worker_work_time_list_upper = []
std_worker_work_time_list_lower = []
mean_worker_supply_list = []
std_worker_supply_list_upper = []
std_worker_supply_list_lower = []

for i in range(max_game_time):
    mean_accumulated_mineral_list.append(
        np.mean(
            [
                (
                    accumulated_mineral[i]
                    if i < len(accumulated_mineral)
                    else accumulated_mineral[-1]
                )
                for accumulated_mineral in all_accumulated_mineral
            ]
        )
    )
    std_accumulated_mineral_list = np.std(
        [
            (
                accumulated_mineral[i]
                if i < len(accumulated_mineral)
                else accumulated_mineral[-1]
            )
            for accumulated_mineral in all_accumulated_mineral
        ]
    )
    std_accumulated_mineral_list_upper.append(
        mean_accumulated_mineral_list[i] + std_accumulated_mineral_list
    )
    std_accumulated_mineral_list_lower.append(
        mean_accumulated_mineral_list[i] - std_accumulated_mineral_list
    )

    mean_accumulated_gas_list.append(
        np.mean(
            [
                (
                    accumulated_gas[i]
                    if i < len(accumulated_gas)
                    else accumulated_gas[-1]
                )
                for accumulated_gas in all_accumulated_gas
            ]
        )
    )
    std_accumulated_gas_list = np.std(
        [
            (accumulated_gas[i] if i < len(accumulated_gas) else accumulated_gas[-1])
            for accumulated_gas in all_accumulated_gas
        ]
    )
    std_accumulated_gas_list_upper.append(
        mean_accumulated_gas_list[i] + std_accumulated_gas_list
    )
    std_accumulated_gas_list_lower.append(
        mean_accumulated_gas_list[i] - std_accumulated_gas_list
    )

    mean_worker_work_time_list.append(
        np.mean(
            [
                (
                    worker_work_time[i]
                    if i < len(worker_work_time)
                    else worker_work_time[-1]
                )
                for worker_work_time in all_worker_work_time
            ]
        )
    )
    std_worker_work_time_list = np.std(
        [
            (worker_work_time[i] if i < len(worker_work_time) else worker_work_time[-1])
            for worker_work_time in all_worker_work_time
        ]
    )
    std_worker_work_time_list_upper.append(
        mean_worker_work_time_list[i] + std_worker_work_time_list
    )
    std_worker_work_time_list_lower.append(
        mean_worker_work_time_list[i] - std_worker_work_time_list
    )

    mean_worker_supply_list.append(
        np.mean(
            [
                (worker_supply[i] if i < len(worker_supply) else worker_supply[-1])
                for worker_supply in all_worker_supply
            ]
        )
    )
    std_worker_supply_list = np.std(
        [
            (worker_supply[i] if i < len(worker_supply) else worker_supply[-1])
            for worker_supply in all_worker_supply
        ]
    )
    std_worker_supply_list_upper.append(
        mean_worker_supply_list[i] + std_worker_supply_list
    )
    std_worker_supply_list_lower.append(
        mean_worker_supply_list[i] - std_worker_supply_list
    )
    
mean_accumulated_mineral_list_baseline = []
std_accumulated_mineral_list_upper_baseline = []
std_accumulated_mineral_list_lower_baseline = []
mean_accumulated_gas_list_baseline = []
std_accumulated_gas_list_upper_baseline = []
std_accumulated_gas_list_lower_baseline = []
mean_worker_work_time_list_baseline = []
std_worker_work_time_list_upper_baseline = []
std_worker_work_time_list_lower_baseline = []
mean_worker_supply_list_baseline = []
std_worker_supply_list_upper_baseline = []
std_worker_supply_list_lower_baseline = []

for i in range(max_game_time_baseline):
    mean_accumulated_mineral_list_baseline.append(
        np.mean(
            [
                (
                    accumulated_mineral[i]
                    if i < len(accumulated_mineral)
                    else accumulated_mineral[-1]
                )
                for accumulated_mineral in all_accumulated_mineral_baseline
            ]
        )
    )
    std_accumulated_mineral_list_baseline = np.std(
        [
            (
                accumulated_mineral[i]
                if i < len(accumulated_mineral)
                else accumulated_mineral[-1]
            )
            for accumulated_mineral in all_accumulated_mineral_baseline
        ]
    )
    std_accumulated_mineral_list_upper_baseline.append(
        mean_accumulated_mineral_list_baseline[i] + std_accumulated_mineral_list_baseline
    )
    std_accumulated_mineral_list_lower_baseline.append(
        mean_accumulated_mineral_list_baseline[i] - std_accumulated_mineral_list_baseline
    )

    mean_accumulated_gas_list_baseline.append(
        np.mean(
            [
                (
                    accumulated_gas[i]
                    if i < len(accumulated_gas)
                    else accumulated_gas[-1]
                )
                for accumulated_gas in all_accumulated_gas_baseline
            ]
        )
    )
    std_accumulated_gas_list_baseline = np.std(
        [
            (accumulated_gas[i] if i < len(accumulated_gas) else accumulated_gas[-1])
            for accumulated_gas in all_accumulated_gas_baseline
        ]
    )
    std_accumulated_gas_list_upper_baseline.append(
        mean_accumulated_gas_list_baseline[i] + std_accumulated_gas_list_baseline
    )
    std_accumulated_gas_list_lower_baseline.append(
        mean_accumulated_gas_list_baseline[i] - std_accumulated_gas_list_baseline
    )

    mean_worker_work_time_list_baseline.append(
        np.mean(
            [
                (
                    worker_work_time[i]
                    if i < len(worker_work_time)
                    else worker_work_time[-1]
                )
                for worker_work_time in all_worker_work_time_baseline
            ]
        )
    )
    std_worker_work_time_list_baseline = np.std(
        [
            (worker_work_time[i] if i < len(worker_work_time) else worker_work_time[-1])
            for worker_work_time in all_worker_work_time_baseline
        ]
    )
    std_worker_work_time_list_upper_baseline.append(
        mean_worker_work_time_list_baseline[i] + std_worker_work_time_list_baseline
    )
    std_worker_work_time_list_lower_baseline.append(
        mean_worker_work_time_list_baseline[i] - std_worker_work_time_list_baseline
    )

    mean_worker_supply_list_baseline.append(
        np.mean(
            [
                (worker_supply[i] if i < len(worker_supply) else worker_supply[-1])
                for worker_supply in all_worker_supply_baseline
            ]
        )
    )
    std_worker_supply_list_baseline = np.std(
        [
            (worker_supply[i] if i < len(worker_supply) else worker_supply[-1])
            for worker_supply in all_worker_supply_baseline
        ]
    )
    std_worker_supply_list_upper_baseline.append(
        mean_worker_supply_list_baseline[i] + std_worker_supply_list_baseline
    )
    std_worker_supply_list_lower_baseline.append(
        mean_worker_supply_list_baseline[i] - std_worker_supply_list_baseline
    )
    
min_max_game_time = min(max_game_time, max_game_time_baseline)

plt.figure(figsize=(12, 8))

# set subplot gap
plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.style.use("science")
plt.subplot(2, 2, 1)
plt.plot(range(min_max_game_time), mean_accumulated_mineral_list[:min_max_game_time], label="Our method")
plt.fill_between(
    range(min_max_game_time),
    std_accumulated_mineral_list_lower[:min_max_game_time],
    std_accumulated_mineral_list_upper[:min_max_game_time],
    alpha=0.2,
)
plt.plot(range(min_max_game_time), mean_accumulated_mineral_list_baseline[:min_max_game_time], label="Baseline")
plt.fill_between(
    range(min_max_game_time),
    std_accumulated_mineral_list_lower_baseline[:min_max_game_time],
    std_accumulated_mineral_list_upper_baseline[:min_max_game_time],
    alpha=0.2,
)
plt.xlabel("Game time (s)")
plt.ylabel("Accumulated mineral storage")
plt.legend(loc="upper left")
plt.title("(a) Accumulated Mineral Storage", y=-0.25, fontdict={"fontsize": 16})

plt.subplot(2, 2, 2)
plt.plot(range(min_max_game_time), mean_accumulated_gas_list[:min_max_game_time], label="Our method")
plt.fill_between(
    range(min_max_game_time),
    std_accumulated_gas_list_lower[:min_max_game_time],
    std_accumulated_gas_list_upper[:min_max_game_time],
    alpha=0.2,
)
plt.plot(range(min_max_game_time), mean_accumulated_gas_list_baseline[:min_max_game_time], label="Baseline")
plt.fill_between(
    range(min_max_game_time),
    std_accumulated_gas_list_lower_baseline[:min_max_game_time],
    std_accumulated_gas_list_upper_baseline[:min_max_game_time],
    alpha=0.2,
)
plt.xlabel("Game time (s)")
plt.ylabel("Accumulated gas storage")
plt.legend(loc="upper left")
plt.title("(b) Accumulated Gas Storage", y=-0.25, fontdict={"fontsize": 16})

plt.subplot(2, 2, 3)
plt.plot(range(min_max_game_time), mean_worker_work_time_list[:min_max_game_time], label="Our method")
plt.fill_between(
    range(min_max_game_time),
    std_worker_work_time_list_lower[:min_max_game_time],
    std_worker_work_time_list_upper[:min_max_game_time],
    alpha=0.2,
)
plt.plot(range(min_max_game_time), mean_worker_work_time_list_baseline[:min_max_game_time], label="Baseline")
plt.fill_between(
    range(min_max_game_time),
    std_worker_work_time_list_lower_baseline[:min_max_game_time],
    std_worker_work_time_list_upper_baseline[:min_max_game_time],
    alpha=0.2,
)
plt.xlabel("Game time (s)")
plt.ylabel("Probe total working time (s)")
plt.legend(loc="upper left")
plt.title("(c) Probe Total Working Time", y=-0.25, fontdict={"fontsize": 16})

plt.subplot(2, 2, 4)
plt.plot(range(min_max_game_time), mean_worker_supply_list[:min_max_game_time], label="Our method")
plt.fill_between(
    range(min_max_game_time),
    std_worker_supply_list_lower[:min_max_game_time],
    std_worker_supply_list_upper[:min_max_game_time],
    alpha=0.2,
)
plt.plot(range(min_max_game_time), mean_worker_supply_list_baseline[:min_max_game_time], label="Baseline")
plt.fill_between(
    range(min_max_game_time),
    std_worker_supply_list_lower_baseline[:min_max_game_time],
    std_worker_supply_list_upper_baseline[:min_max_game_time],
    alpha=0.2,
)
plt.xlabel("Game time (s)")
plt.ylabel("Probe supply")
plt.legend(loc="upper left")
plt.title("(d) Probe Supply", y=-0.25, fontdict={"fontsize": 16})

plt.savefig(".\\sc2_rl_agent\\starcraftenv_test\\log\\chatgpt_log\\img\\apendix\\4in1_d3.pdf")
# plt.show()
