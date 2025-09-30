from colorama import Fore

a = float(input("a (Pu): "))
b = float(input("b (Pv): "))
Du = float(input("Du: "))
Dv = float(input("Dv: "))

val1 = -1 + 2 * b / (a + b) - (a + b)**2
cond1 = val1 < 0
color = Fore.GREEN if cond1 else Fore.RED
print("f_u + g_v < 0:\t\t\t\t\t" + color + str(cond1) + Fore.WHITE + f"\t\t (f_u + g_v = {val1})")

val2 = Dv/Du * (-1 + 2 * b / (a + b)) - (a + b)**2
cond2 = val2 > 0
color = Fore.GREEN if cond2 else Fore.RED
print("df_u + g_v > 0:\t\t\t\t\t" + color + str(cond2) + Fore.WHITE + f"\t\t (df_u + g_v = {val2})")

val3 = (a + b)**2
cond3 = val3 > 0
color = Fore.GREEN if cond3 else Fore.RED
print("f_u * g_v - f_v * g_u > 0:\t\t\t" + color + str(cond3) + Fore.WHITE + f"\t\t (f_u * g_v - f_v * g_u = {val3})")

val4 = (Dv/Du * (-1 + 2 * b / (a + b)) - (a + b)**2)**2 - 4 * Dv/Du * (a + b)**2
cond4 = val4 > 0
color = Fore.GREEN if cond4 else Fore.RED
print("(df_u + g_v)^2 - 4d(f_u * g_v - f_v * g_u) > 0:\t" + color + str(cond4) + Fore.WHITE + f"\t\t (df_u + g_v)^2 - 4d(f_u * g_v - f_v * g_u) = {val4})")