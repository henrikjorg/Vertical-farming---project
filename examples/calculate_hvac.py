# Vil ha:
T_sup = 22.3
Chi_sup = 17

# Har:
T_in = 24
Chi_in = 18

T_out = 14
Chi_out = 10

eta_rot_T = 0.8
eta_rot_Chi = 0.8
u_rot = 1
T_rot = u_rot*eta_rot_T*(T_in - T_out) + T_out
Chi_rot = u_rot*eta_rot_Chi*(Chi_in - Chi_out) + Chi_out

print("T_rot: ", T_rot)
print("Chi_rot: ", Chi_rot)