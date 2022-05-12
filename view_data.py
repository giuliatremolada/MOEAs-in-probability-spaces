import json

f_parego = open("Risultati\\parego_data.json",)
parego_data = json.load(f_parego)
f_nsga = open("Risultati\\nsga_data.json",)
nsga_data = json.load(f_nsga)

#ogni trial va da 0 a iter-1 (scalando con + 20 * trial in base al trial che si vuole)
print(parego_data[16 + (20*3)], "\n", parego_data[17 + (20*3)], "\n", parego_data[18 + (20*3)], "\n", parego_data[19 + (20*3)])
