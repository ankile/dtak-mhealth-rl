import tkinter as tk
from tkinter import ttk


def save_to_file():
    with open("rewards_matrix.txt", "w") as f:
        for row in gridworld:
            row_values = [str(cell.get()) for cell in row]
            f.write(",".join(row_values) + "\n")


def create_gridworld():
    n, m = int(n_var.get()), int(m_var.get())

    for row in gridworld:
        for cell in row:
            cell.grid_forget()

    for i in range(n):
        row = []
        for j in range(m):
            cell_var = tk.StringVar(value="0")
            cell = ttk.Entry(main_frame, textvariable=cell_var, width=5)
            cell.grid(row=i, column=j, padx=5, pady=5)
            row.append(cell_var)
        gridworld.append(row)


root = tk.Tk()
root.title("Gridworld Rewards Matrix")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0)

n_label = ttk.Label(main_frame, text="N:")
n_label.grid(row=0, column=0)

n_var = tk.StringVar()
n_entry = ttk.Entry(main_frame, textvariable=n_var, width=5)
n_entry.grid(row=0, column=1)

m_label = ttk.Label(main_frame, text="M:")
m_label.grid(row=0, column=2)

m_var = tk.StringVar()
m_entry = ttk.Entry(main_frame, textvariable=m_var, width=5)
m_entry.grid(row=0, column=3)

create_button = ttk.Button(main_frame, text="Create Grid", command=create_gridworld)
create_button.grid(row=0, column=4, padx=10)

save_button = ttk.Button(main_frame, text="Export Rewards", command=save_to_file)
save_button.grid(row=0, column=5)

gridworld = []

root.mainloop()
