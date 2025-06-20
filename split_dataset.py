import pickle
import os

input_file = 'supervised_pbs_dataset_partialorderlabels.pkl'   # your big pickle file
output_folder = 'batched_dataset_1000'

os.makedirs(output_folder, exist_ok=True)

with open(input_file, 'rb') as f:
    data = pickle.load(f)

batch_size = 1000
total = len(data)
num_batches = (total + batch_size - 1) // batch_size

print(f"Splitting {total} samples into {num_batches} batch files...")

for i in range(num_batches):
    batch = data[i*batch_size : (i+1)*batch_size]
    batch_file = os.path.join(output_folder, f'batch_{i:05d}.pkl')
    with open(batch_file, 'wb') as f:
        pickle.dump(batch, f)
    print(f"Wrote {batch_file} ({len(batch)} samples)")

print("Done!")