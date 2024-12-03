# for 1D2 and 1P1
S1 = 0; S2 = 0
L1 = 2; L2 = 1
J1 = 2; J2 = 1

# ev =
# hbar =
# me =

gj1 = gJ(S1,J1,L1)
gj2 = gJ(S2,J2,L2)


def gJ(S,J,L):
    return = 3/2 + (S*(S+1)-L*(L+1) )/(2*J*(J+1))


def deltaE(Bfield, gj1, gj2, Mj1,Mj2):
    return ev * hbar / 2 / me * Bfield * (gj1*Mj1 + gj2 * Mj2)


import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

# Calibration of the B field OLD SETUP
# Amps = [0.98,1.4,1.98,2.55,3.07,3.43,4.11,4.54,5.17,5.69,6.03,6.55,7.10,7.56,8.08,8.55,9.04]
# Bfield = [-57,-86.8,-119.5,-158.4,-190,-216,-261,-291,-332,-364,-387,-418,-455,-483,-513,-537,-562] # +- 2 mT

# NEW setup B-FIELD !!!
Amps = [0,1.25,1.98,3.14,4.04,5.14,6.15,7.06,8.07,9.03]
Bfield = [0.58,76.7,123,195,248,314,375,430,492,534] #=-1


plt.figure()
plt.grid()
plt.plot(Amps,-1*np.array(Bfield),'-o')
plt.title("Magnetic field vs. Input Current Calibration")
plt.xlabel("Input Curret (A)")
plt.ylabel("B-field (mT)")
plt.savefig('bfieldcal.png')


# CIRCLES!
# B-field = 7.5 +- 0.1
# perims +- 0.5
164.9
372.3

650.3 #3.1
732.0 #3.2
799.5 #3.3







# Load the .dat file
with open('907A-p90.dat', 'r') as f:  # Replace 'file.dat' with your actual file path
    hex_data = f.read().split()

# Convert hex to integers
int_data = [int(value, 16) for value in hex_data]

# # Determine the dimensions for reshaping (example: make it square or choose a custom shape)
# side_length = int(len(int_data) ** 0.5)
# data_2d = np.array(int_data[:side_length**2]).reshape(side_length, side_length)

data_2d = np.array(int_data).reshape(1024,320)

# Increase contrast by setting vmin and vmax
vmin = np.percentile(data_2d, 5)  # Lower 5th percentile
vmax = np.percentile(data_2d, 95)  # Upper 95th percentile

# Plot the data
plt.figure(figsize=(8, 6))
plt.imshow(data_2d, cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(label='Intensity')
plt.title('Enhanced Contrast Image')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


#########INVERTING

import pandas as pd

# Read the Excel file into a pandas DataFrame
input_file = '113A-p0.xls'  # Replace with your file path
df = pd.read_csv(input_file, sep='\t')

# Invert the DataFrame (transpose it)
# inverted_df = df.transpose()
flipped_df = df.iloc[::-1].reset_index(drop=True)  # Reverse rows and reset the index
flipped_df['X'] = flipped_df.index  # Assign the reversed index as x-values


# Save the inverted DataFrame to a new Excel file
output_file = '113A-p0-inverted.xls'  # Replace with your desired output file path
flipped_df.to_csv(output_file, index=False, sep='\t')

print(f"Inverted array saved to {output_file}")




###############################
    
df = pd.read_csv('peaks.csv')


inds = df['pangle'] == 0
amps = df['amps'][inds]

dists01 = [json.loads(df['peaks'][inds][i])[0]-df['xcenter'][inds][i] for i in df.index[inds]]
dists12 = [json.loads(df['peaks'][inds][i])[1]-json.loads(df['peaks'][inds][i])[0] for i in df.index[inds]]
dists23 = [json.loads(df['peaks'][inds][i])[2]-json.loads(df['peaks'][inds][i])[1] for i in df.index[inds]]

plt.figure()
plt.plot(amps,dists01,'o',label = '0-1')
plt.plot(amps,dists12,'o',label = '1-2')
plt.plot(amps,dists23,'o',label = '2-3')

plt.xlabel("Voltage (Amps)")
plt.ylabel("Distance to ring (pixels)")
plt.grid()
plt.legend()
plt.show()