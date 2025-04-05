import tkinter as tk
from tkinter import messagebox
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Declarar símbolo para sympy
x = sp.symbols('x')

def parse_function(expr):
    try:
        # Convierte la cadena de entrada en una expresión simbólica
        func_expr = sp.sympify(expr)
        return func_expr
    except sp.SympifyError:
        return None

def falsa_posicion(func_expr, a, b, tol, max_iter):
    fa = float(func_expr.subs(x, a))
    fb = float(func_expr.subs(x, b))
    if fa * fb >= 0:
        raise ValueError("La función no cambia de signo en el intervalo [a, b].")
    iterations = []
    for i in range(max_iter):
        # Fórmula de la falsa posición
        c = (a * fb - b * fa) / (fb - fa)
        fc = float(func_expr.subs(x, c))
        iterations.append((i+1, a, b, c, fc))
        if abs(fc) < tol:
            return c, i + 1, iterations
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, max_iter, iterations

def newton_raphson(func_expr, x0, tol, max_iter):
    f_val = lambda val: float(func_expr.subs(x, val))
    # Derivada simbólica de la función
    dfunc_expr = sp.diff(func_expr, x)
    df_val = lambda val: float(dfunc_expr.subs(x, val))
    
    if df_val(x0) == 0:
        raise ValueError("La derivada es cero en el valor inicial. Newton-Raphson no puede aplicarse.")
    
    iterations = []
    xi = x0
    for i in range(max_iter):
        dfxi = df_val(xi)
        if dfxi == 0:
            raise ValueError("Se encontró derivada cero durante la iteración.")
        xi_new = xi - f_val(xi) / dfxi
        error = abs(xi_new - xi)
        iterations.append((i+1, xi, f_val(xi), xi_new, error))
        if error < tol:
            return xi_new, i + 1, iterations
        xi = xi_new
    return xi, max_iter, iterations

class RootFinderApp:
    def __init__(self, master):
        self.master = master
        master.title("Calculador de Raíces - Métodos Separados")
        
        # Configuración de la ventana principal para que sea responsiva
        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.rowconfigure(1, weight=1)

        # Frame común para la función, tolerancia y max iteraciones
        self.common_frame = tk.Frame(master)
        self.common_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        self.common_frame.columnconfigure(1, weight=1)
        
        tk.Label(self.common_frame, text="Función f(x):").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.func_entry = tk.Entry(self.common_frame, width=30)
        self.func_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        tk.Label(self.common_frame, text="Tolerancia:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.tol_entry = tk.Entry(self.common_frame, width=15)
        self.tol_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        tk.Label(self.common_frame, text="Máx iteraciones:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.iter_entry = tk.Entry(self.common_frame, width=15)
        self.iter_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Frame para el método de Falsa Posición
        self.frame_fp = tk.LabelFrame(master, text="Método de Falsa Posición", padx=10, pady=10)
        self.frame_fp.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.frame_fp.columnconfigure(1, weight=1)
        
        tk.Label(self.frame_fp, text="Valor a:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.a_entry = tk.Entry(self.frame_fp, width=15)
        self.a_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        tk.Label(self.frame_fp, text="Valor b:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.b_entry = tk.Entry(self.frame_fp, width=15)
        self.b_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        self.calc_fp_button = tk.Button(self.frame_fp, text="Calcular Falsa Posición", command=self.calculate_fp)
        self.calc_fp_button.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
        
        self.result_fp_text = tk.Text(self.frame_fp, height=10, width=40)
        self.result_fp_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # Frame para el método de Newton-Raphson
        self.frame_nr = tk.LabelFrame(master, text="Método de Newton-Raphson", padx=10, pady=10)
        self.frame_nr.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.frame_nr.columnconfigure(1, weight=1)
        
        tk.Label(self.frame_nr, text="Valor x0:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.x0_entry = tk.Entry(self.frame_nr, width=15)
        self.x0_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        self.calc_nr_button = tk.Button(self.frame_nr, text="Calcular Newton-Raphson", command=self.calculate_nr)
        self.calc_nr_button.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
        
        self.result_nr_text = tk.Text(self.frame_nr, height=10, width=40)
        self.result_nr_text.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
    def calculate_fp(self):
        self.result_fp_text.delete("1.0", tk.END)
        func_str = self.func_entry.get()
        func_expr = parse_function(func_str)
        if func_expr is None:
            messagebox.showwarning("Error", "La función ingresada no es válida.")
            return
        
        try:
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            tol = float(self.tol_entry.get())
            max_iter = int(self.iter_entry.get())
        except ValueError:
            messagebox.showwarning("Error", "Por favor, ingrese valores numéricos correctos en los campos comunes o de Falsa Posición.")
            return
        
        # Validar condiciones para Falsa Posición
        fa = float(func_expr.subs(x, a))
        fb = float(func_expr.subs(x, b))
        if fa * fb >= 0:
            messagebox.showwarning("Error", "Falsa Posición: La función no cambia de signo en [a, b]. Verifique los valores.")
            return
        
        try:
            raiz_fp, iter_fp, iter_data = falsa_posicion(func_expr, a, b, tol, max_iter)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        
        resultado = f"Resultados:\nRaíz aproximada: {raiz_fp:.6f}\nIteraciones: {iter_fp}\n\n"
        resultado += "Detalles de iteración:\n"
        for data in iter_data:
            # data: (iteración, a, b, c, f(c))
            resultado += (f"Iteración {data[0]}: a = {data[1]:.6f}, b = {data[2]:.6f}, "
                          f"c = {data[3]:.6f}, f(c) = {data[4]:.6e}\n")
        self.result_fp_text.insert(tk.END, resultado)
        # Mostrar la gráfica para Falsa Posición usando el intervalo [a, b]
        self.graficar_fp(func_expr, a, b, raiz_fp)
    
    def calculate_nr(self):
        self.result_nr_text.delete("1.0", tk.END)
        func_str = self.func_entry.get()
        func_expr = parse_function(func_str)
        if func_expr is None:
            messagebox.showwarning("Error", "La función ingresada no es válida.")
            return
        
        try:
            x0 = float(self.x0_entry.get())
            tol = float(self.tol_entry.get())
            max_iter = int(self.iter_entry.get())
        except ValueError:
            messagebox.showwarning("Error", "Por favor, ingrese valores numéricos correctos en los campos comunes o de Newton-Raphson.")
            return
        
        # Validar condición para Newton-Raphson
        dfunc_expr = sp.diff(func_expr, x)
        if float(dfunc_expr.subs(x, x0)) == 0:
            messagebox.showwarning("Error", "Newton-Raphson: La derivada es cero en x0. Ingrese otro valor inicial.")
            return
        
        try:
            raiz_nr, iter_nr, iter_data = newton_raphson(func_expr, x0, tol, max_iter)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        
        resultado = f"Resultados:\nRaíz aproximada: {raiz_nr:.6f}\nIteraciones: {iter_nr}\n\n"
        resultado += "Detalles de iteración:\n"
        for data in iter_data:
            # data: (iteración, xi, f(xi), xi_new, error)
            resultado += (f"Iteración {data[0]}: xi = {data[1]:.6f}, f(xi) = {data[2]:.6e}, "
                          f"xi_new = {data[3]:.6f}, error = {data[4]:.6e}\n")
        self.result_nr_text.insert(tk.END, resultado)
        # Para Newton-Raphson, se genera un intervalo basado en x0 y la raíz encontrada
        self.graficar_nr(func_expr, x0, raiz_nr)
        
    def graficar_fp(self, func_expr, a, b, raiz_fp):
        xs = np.linspace(a, b, 400)
        f_lambd = sp.lambdify(x, func_expr, "numpy")
        ys = f_lambd(xs)
        
        plt.figure()
        plt.plot(xs, ys, label="f(x)")
        plt.axhline(0, color="black", linewidth=0.5)
        plt.plot(raiz_fp, f_lambd(raiz_fp), "ro", label="Falsa Posición")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Gráfica (Falsa Posición)")
        plt.show()
    
    def graficar_nr(self, func_expr, x0, raiz_nr):
        # Definir un intervalo basado en x0 y la raíz obtenida
        delta = abs(x0 - raiz_nr)
        if delta == 0:
            delta = 1
        lower = min(x0, raiz_nr) - 2*delta
        upper = max(x0, raiz_nr) + 2*delta
        xs = np.linspace(lower, upper, 400)
        f_lambd = sp.lambdify(x, func_expr, "numpy")
        ys = f_lambd(xs)
        
        plt.figure()
        plt.plot(xs, ys, label="f(x)")
        plt.axhline(0, color="black", linewidth=0.5)
        plt.plot(raiz_nr, f_lambd(raiz_nr), "go", label="Newton-Raphson")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Gráfica (Newton-Raphson)")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    # Configuración de la ventana principal para que se expanda responsivamente
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(1, weight=1)
    app = RootFinderApp(root)
    root.mainloop()
