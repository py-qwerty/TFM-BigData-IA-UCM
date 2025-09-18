

import cvxpy as cp
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import make_interp_spline

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)


# Variable global para almacenar configuraciones
configuraciones_guardadas = []

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def plot_promedio_horas_por_bloque(configuraciones_guardadas, topic_blocks):
    """
    Muestra el promedio de horas asignadas por bloque para cada configuración.
    """
    if not configuraciones_guardadas:
        logger.warning("No hay configuraciones guardadas para analizar.")
        return

    # Preparar datos
    datos_bloques = []

    for config in configuraciones_guardadas:
        nombre_config = config['nombre']
        temas = config['temas']
        horas = config['horas']

        # Agrupar por bloques
        horas_b1 = []
        horas_b2 = []
        horas_b3 = []

        for i, tema in enumerate(temas):
            if tema in topic_blocks:
                bloque = topic_blocks[tema]
                if bloque == 1:
                    horas_b1.append(horas[i])
                elif bloque == 2:
                    horas_b2.append(horas[i])
                elif bloque == 3:
                    horas_b3.append(horas[i])

        # Calcular promedios
        promedio_b1 = np.mean(horas_b1) if horas_b1 else 0
        promedio_b2 = np.mean(horas_b2) if horas_b2 else 0
        promedio_b3 = np.mean(horas_b3) if horas_b3 else 0

        datos_bloques.append({
            'Configuración': nombre_config,
            'Bloque 1': promedio_b1,
            'Bloque 2': promedio_b2,
            'Bloque 3': promedio_b3
        })

    # Crear DataFrame
    df = pd.DataFrame(datos_bloques)

    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("Set2")

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('white')

    # Posiciones de las barras
    x = np.arange(len(df))
    width = 0.25

    # Colores distintivos para cada bloque
    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    # Crear barras agrupadas
    bars1 = ax.bar(x - width, df['Bloque 1'], width, label='Bloque 1',
                   color=colores[0], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x, df['Bloque 2'], width, label='Bloque 2',
                   color=colores[1], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars3 = ax.bar(x + width, df['Bloque 3'], width, label='Bloque 3',
                   color=colores[2], alpha=0.8, edgecolor='white', linewidth=1.5)

    # Añadir valores encima de las barras
    def añadir_valores(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}h',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold',
                            color='#2c3e50')

    añadir_valores(bars1)
    añadir_valores(bars2)
    añadir_valores(bars3)

    # Configurar ejes y títulos
    ax.set_xlabel('Configuraciones', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Promedio de Horas por Tema', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_title('Comparación de Promedio de Horas por Bloque entre Configuraciones',
                 fontsize=16, fontweight='bold', pad=20, color='#2c3e50')

    # Configurar etiquetas del eje X
    ax.set_xticks(x)
    ax.set_xticklabels(df['Configuración'], rotation=45, ha='right', fontsize=11)

    # Leyenda
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                       fancybox=True, shadow=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)

    # Ajustar layout
    plt.tight_layout()

    # Estadísticas resumidas
    total_configs = len(df)
    promedio_general_b1 = df['Bloque 1'].mean()
    promedio_general_b2 = df['Bloque 2'].mean()
    promedio_general_b3 = df['Bloque 3'].mean()

    stats_text = (f'Estadísticas Generales:\n'
                  f'• Configuraciones: {total_configs}\n'
                  f'• Prom. B1: {promedio_general_b1:.1f}h\n'
                  f'• Prom. B2: {promedio_general_b2:.1f}h\n'
                  f'• Prom. B3: {promedio_general_b3:.1f}h')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            fontweight='bold', bbox=dict(boxstyle="round,pad=0.6",
                                         facecolor='lightblue', alpha=0.9,
                                         edgecolor='navy'),
            verticalalignment='top')

    plt.show()
    logger.info(f"Gráfico de promedios por bloque mostrado para {total_configs} configuraciones")


def plot_comparacion_rendimiento_horas(configuraciones_guardadas, tema_especifico=None):
    """
    Compara rendimiento (theta) y horas asignadas con gráficas superpuestas.
    Si se especifica tema_especifico, muestra solo ese tema. Si no, muestra todos.
    """
    if not configuraciones_guardadas:
        logger.warning("No hay configuraciones guardadas para comparar.")
        return

    # Configurar estilo
    plt.style.use('default')
    sns.set_style("whitegrid", {"grid.linewidth": 0.5, "grid.alpha": 0.3})

    # Crear figura
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('white')

    # Colores para horas y theta
    color_horas = '#2E86AB'  # Azul para horas
    color_theta = '#F24236'  # Rojo para theta

    if tema_especifico:
        # Modo: tema específico
        configuraciones_nombres = []
        valores_horas = []
        valores_theta = []

        for config in configuraciones_guardadas:
            if tema_especifico in config['temas']:
                idx = config['temas'].index(tema_especifico)
                configuraciones_nombres.append(config['nombre'])
                valores_horas.append(config['horas'][idx])
                valores_theta.append(config['thetas'][idx])

        if not configuraciones_nombres:
            logger.warning(f"El tema {tema_especifico} no se encuentra en ninguna configuración.")
            return

        x_pos = np.arange(len(configuraciones_nombres))

        # Normalizar theta para comparación visual
        max_horas = max(valores_horas) if valores_horas else 1
        max_theta = max(valores_theta) if valores_theta else 1
        factor_escala = max_horas / max_theta if max_theta > 0 else 1
        theta_escalado = [t * factor_escala for t in valores_theta]

        # Crear barras superpuestas
        bars_horas = ax.bar(x_pos, valores_horas, alpha=0.7, color=color_horas,
                            label=f'Horas Asignadas (Tema {tema_especifico})',
                            edgecolor='white', linewidth=2)

        bars_theta = ax.bar(x_pos, theta_escalado, alpha=0.6, color=color_theta,
                            label=f'Rendimiento Escalado (×{factor_escala:.1f})',
                            edgecolor='white', linewidth=2)

        # Etiquetas de valores
        for i, (h, t_orig, t_esc) in enumerate(zip(valores_horas, valores_theta, theta_escalado)):
            # Etiqueta de horas
            ax.annotate(f'{h}h', xy=(i, h), xytext=(0, 5),
                        textcoords="offset points", ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color=color_horas)

            # Etiqueta de theta original
            ax.annotate(f'θ={t_orig:.3f}', xy=(i, t_esc), xytext=(0, -20),
                        textcoords="offset points", ha='center', va='top',
                        fontsize=10, fontweight='bold', color=color_theta)

        ax.set_xlabel('Configuraciones', fontsize=14, fontweight='bold')
        ax.set_ylabel('Horas / Theta Escalado', fontsize=14, fontweight='bold')
        ax.set_title(f'Comparación Rendimiento vs Horas Asignadas - Tema {tema_especifico}',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configuraciones_nombres, rotation=45, ha='right')

    else:
        # Modo: todos los temas superpuestos
        todos_los_temas = set()
        for config in configuraciones_guardadas:
            todos_los_temas.update(config['temas'])
        todos_los_temas = sorted(list(todos_los_temas))

        # Calcular factor de escala global
        max_horas_global = 0
        max_theta_global = 0

        for config in configuraciones_guardadas:
            max_horas_global = max(max_horas_global, max(config['horas']) if config['horas'] else 0)
            max_theta_global = max(max_theta_global, max(config['thetas']) if config['thetas'] else 0)

        factor_escala_global = max_horas_global / max_theta_global if max_theta_global > 0 else 1

        # Transparencias diferenciadas
        alpha_horas = 0.6
        alpha_theta = 0.5

        # Plotear cada configuración
        for i, config in enumerate(configuraciones_guardadas):
            temas = config['temas']
            horas = config['horas']
            thetas = config['thetas']

            # Escalar theta
            thetas_escalados = [t * factor_escala_global for t in thetas]

            # Color único para cada configuración pero manteniendo familia de colores
            color_base_horas = plt.cm.Blues(0.5 + 0.3 * (i / len(configuraciones_guardadas)))
            color_base_theta = plt.cm.Reds(0.5 + 0.3 * (i / len(configuraciones_guardadas)))

            # Líneas y áreas
            if len(temas) > 1:
                # Área bajo la curva para horas
                ax.fill_between(temas, 0, horas, alpha=alpha_horas * 0.3,
                                color=color_base_horas, label=f'{config["nombre"]} - Horas (área)')

                # Área bajo la curva para theta
                ax.fill_between(temas, 0, thetas_escalados, alpha=alpha_theta * 0.3,
                                color=color_base_theta, label=f'{config["nombre"]} - Theta (área)')

                # Líneas principales
                ax.plot(temas, horas, '-', color=color_base_horas, linewidth=3,
                        alpha=alpha_horas + 0.3, label=f'{config["nombre"]} - Horas')

                ax.plot(temas, thetas_escalados, '--', color=color_base_theta, linewidth=2.5,
                        alpha=alpha_theta + 0.3, label=f'{config["nombre"]} - Theta')

            # Puntos
            ax.scatter(temas, horas, color=color_base_horas, s=80, alpha=alpha_horas + 0.3,
                       edgecolors='white', linewidth=1.5, zorder=5)

            ax.scatter(temas, thetas_escalados, color=color_base_theta, s=60, alpha=alpha_theta + 0.3,
                       marker='s', edgecolors='white', linewidth=1.5, zorder=5)

        ax.set_xlabel('Número de Tema', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Horas / Theta Escalado (×{factor_escala_global:.1f})', fontsize=14, fontweight='bold')
        ax.set_title('Comparación Superpuesta: Rendimiento vs Horas Asignadas (Todas las Configuraciones)',
                     fontsize=16, fontweight='bold', pad=20)

        # Configurar límites del eje X
        if todos_los_temas:
            ax.set_xlim(min(todos_los_temas) - 1, max(todos_los_temas) + 1)

    # Configuración común
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)

    # Leyenda
    if tema_especifico:
        legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                           fancybox=True, shadow=True, framealpha=0.95)
    else:
        # Para múltiples configuraciones, leyenda más compacta
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
                           frameon=True, fancybox=True, shadow=True, framealpha=0.95)

    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')

    # Información estadística
    if tema_especifico:
        correlacion = np.corrcoef(valores_horas, valores_theta)[0, 1] if len(valores_horas) > 1 else 0
        stats_text = (f'Estadísticas Tema {tema_especifico}:\n'
                      f'• Configs analizadas: {len(configuraciones_nombres)}\n'
                      f'• Correlación H-θ: {correlacion:.3f}\n'
                      f'• Factor escala: {factor_escala:.2f}')
    else:
        total_puntos = sum(len(config['temas']) for config in configuraciones_guardadas)
        stats_text = (f'Estadísticas Generales:\n'
                      f'• Configuraciones: {len(configuraciones_guardadas)}\n'
                      f'• Total puntos: {total_puntos}\n'
                      f'• Factor escala: {factor_escala_global:.2f}\n'
                      f'• Superposición activa')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            fontweight='bold', bbox=dict(boxstyle="round,pad=0.6",
                                         facecolor='lightyellow', alpha=0.9,
                                         edgecolor='orange'),
            verticalalignment='top')

    # Ajustar layout
    if not tema_especifico:
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Hacer espacio para leyenda externa
    else:
        plt.tight_layout()

    plt.show()

    if tema_especifico:
        logger.info(f"Gráfico de comparación mostrado para tema {tema_especifico}")
    else:
        logger.info(
            f"Gráfico de comparación superpuesta mostrado para {len(configuraciones_guardadas)} configuraciones")


# Función de conveniencia para usar plot_comparacion_rendimiento_horas con todos los temas
def plot_superposicion_rendimiento_horas():
    """
    Wrapper para mostrar la comparación superpuesta de todas las configuraciones.
    """
    plot_comparacion_rendimiento_horas(configuraciones_guardadas)


# Función de conveniencia para analizar un tema específico
def plot_tema_especifico(tema):
    """
    Wrapper para mostrar la comparación de un tema específico.
    """
    plot_comparacion_rendimiento_horas(configuraciones_guardadas, tema_especifico=tema)


def plot_and_save_results(resultado: Dict[str, int], omega: Dict[int, float], theta: Dict[int, float],
                          nombre_config: str = None, mostrar_ahora: bool = True):
    """
    Genera y muestra un gráfico de líneas y puntos con los resultados de la asignación en Jupyter.
    Incluye theta escalado para comparación visual.
    """
    global configuraciones_guardadas

    if not resultado:
        logger.warning("No hay resultados que graficar.")
        return

    # Preparar datos para el gráfico
    temas = sorted([int(k) for k in resultado.keys()])
    horas = [resultado[str(t)] for t in temas]
    importancias = [omega.get(t, 0) for t in temas]
    thetas = [theta.get(t, 0.1) for t in temas]

    # Guardar configuración
    config = {
        'nombre': nombre_config or f"Configuración {len(configuraciones_guardadas) + 1}",
        'temas': temas,
        'horas': horas,
        'importancias': importancias,
        'thetas': thetas
    }
    configuraciones_guardadas.append(config)

    if mostrar_ahora:
        plot_comparacion_configuraciones()


def plot_comparacion_configuraciones_suavizado():
    """
    Muestra todas las configuraciones guardadas con líneas suavizadas
    cubriendo todo el rango de temas del 1 al 45.
    """
    global configuraciones_guardadas

    if not configuraciones_guardadas:
        logger.warning("No hay configuraciones guardadas para comparar.")
        return

    # Configurar estilo elegante
    plt.style.use('default')
    sns.set_palette("husl")
    sns.set_style("whitegrid", {
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3
    })

    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16))
    fig.patch.set_facecolor('#fafafa')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # RANGO COMPLETO de temas del 1 al 45
    todos_los_temas = np.array(range(1, 46))

    # Crear matrices completas para cada configuración
    configuraciones_completas = {}

    for config in configuraciones_guardadas:
        horas_completas = np.zeros(45)
        theta_completas = np.full(45, 0.1)

        for i, tema in enumerate(config['temas']):
            if 1 <= tema <= 45:
                horas_completas[tema - 1] = config['horas'][i]
                theta_completas[tema - 1] = config['thetas'][i]

        configuraciones_completas[config['nombre']] = {
            'horas': horas_completas,
            'theta': theta_completas
        }

    # Colores distintivos para cada configuración
    colores = [
        '#2C3E50',  # Azul oscuro
        '#E74C3C',  # Rojo brillante
        '#3498DB',  # Azul cielo
        '#2ECC71',  # Verde esmeralda
        '#F39C12',  # Naranja
        '#9B59B6',  # Púrpura
        '#1ABC9C',  # Turquesa
        '#34495E',  # Gris oscuro
        '#8E44AD',  # Violeta
        '#D35400'  # Naranja oscuro
    ]

    # Calcular factor de escalado para theta
    max_horas_global = max(np.max(datos['horas']) for datos in configuraciones_completas.values())
    max_theta_global = max(np.max(datos['theta']) for datos in configuraciones_completas.values())
    theta_escalado_factor = max_horas_global / max_theta_global if max_theta_global > 0 else 1

    # GRÁFICO 1: LÍNEAS SUAVIZADAS PARA HORAS
    for i, (nombre_config, datos) in enumerate(configuraciones_completas.items()):
        color = colores[i % len(colores)]
        horas_valores = datos['horas']

        # Encontrar puntos con datos (no cero)
        indices_con_datos = np.where(horas_valores > 0)[0]

        if len(indices_con_datos) > 0:
            # Usar todos los temas para interpolación suave
            x_data = todos_los_temas
            y_data = horas_valores

            # Crear curva suavizada
            x_smooth = np.linspace(1, 45, 200)

            if len(indices_con_datos) >= 4:  # Suficientes puntos para spline cúbico
                try:
                    # Spline cúbico para suavizado elegante
                    spl = make_interp_spline(x_data, y_data, k=3)
                    y_smooth = spl(x_smooth)
                    y_smooth = np.maximum(y_smooth, 0)  # No valores negativos

                    # ÁREA SOMBREADA bajo la curva
                    ax1.fill_between(x_smooth, 0, y_smooth, color=color, alpha=0.15, zorder=1)

                    # LÍNEA SUAVIZADA principal
                    ax1.plot(x_smooth, y_smooth, '-', color=color, linewidth=4,
                             alpha=0.9, zorder=3, label=nombre_config)

                except Exception:
                    # Fallback a interpolación lineal
                    from scipy.interpolate import interp1d
                    f = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value=0)
                    y_smooth = f(x_smooth)
                    ax1.plot(x_smooth, y_smooth, '-', color=color, linewidth=3, alpha=0.8, label=nombre_config)

            else:
                # Pocos puntos: usar línea simple
                ax1.plot(x_data, y_data, '-', color=color, linewidth=3, alpha=0.8, label=nombre_config)

            # PUNTOS DE DATOS ORIGINALES (solo los que tienen valor)
            puntos_x = todos_los_temas[indices_con_datos]
            puntos_y = horas_valores[indices_con_datos]

            # Sombra de puntos
            ax1.scatter(puntos_x, puntos_y, color='black', s=120, alpha=0.2, zorder=4)

            # Puntos principales
            ax1.scatter(puntos_x, puntos_y, color=color, s=100, zorder=5,
                        edgecolors='white', linewidth=2.5, alpha=0.95)

    # GRÁFICO 2: LÍNEAS SUAVIZADAS PARA THETA ESCALADO
    for i, (nombre_config, datos) in enumerate(configuraciones_completas.items()):
        color = colores[i % len(colores)]
        theta_valores = datos['theta'] * theta_escalado_factor

        # Encontrar puntos con datos significativos
        indices_con_datos = np.where(datos['horas'] > 0)[0]  # Basado en donde hay horas asignadas

        if len(indices_con_datos) > 0:
            x_data = todos_los_temas
            y_data = theta_valores

            # Crear curva suavizada
            x_smooth = np.linspace(1, 45, 200)

            if len(indices_con_datos) >= 4:
                try:
                    # Spline para theta
                    spl_theta = make_interp_spline(x_data, y_data, k=3)
                    y_smooth_theta = spl_theta(x_smooth)
                    y_smooth_theta = np.maximum(y_smooth_theta, 0)

                    # ÁREA SOMBREADA para theta
                    ax2.fill_between(x_smooth, 0, y_smooth_theta, color=color, alpha=0.12, zorder=1)

                    # LÍNEA SUAVIZADA con patrón diferente
                    ax2.plot(x_smooth, y_smooth_theta, '--', color=color, linewidth=3.5,
                             alpha=0.85, zorder=3, label=f'{nombre_config} - Theta')

                except Exception:
                    from scipy.interpolate import interp1d
                    f = interp1d(x_data, y_data, kind='linear', bounds_error=False,
                                 fill_value=0.1 * theta_escalado_factor)
                    y_smooth_theta = f(x_smooth)
                    ax2.plot(x_smooth, y_smooth_theta, '--', color=color, linewidth=3, alpha=0.8,
                             label=f'{nombre_config} - Theta')
            else:
                ax2.plot(x_data, y_data, '--', color=color, linewidth=3, alpha=0.8,
                         label=f'{nombre_config} - Theta')

            # PUNTOS para theta (cuadrados)
            puntos_x = todos_los_temas[indices_con_datos]
            puntos_y = theta_valores[indices_con_datos]

            ax2.scatter(puntos_x, puntos_y, color=color, s=80, zorder=5,
                        edgecolors='white', linewidth=2, alpha=0.9, marker='s')

    # CONFIGURACIÓN DEL GRÁFICO 1 (HORAS)
    ax1.set_xlabel("Número de Tema", fontsize=15, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel("Horas Asignadas", fontsize=15, fontweight='bold', color='#2c3e50')
    ax1.set_title("📊 Comparación Suavizada: Distribución de Horas por Configuración (Temas 1-45)",
                  fontsize=17, fontweight='bold', pad=25, color='#2c3e50')

    # CONFIGURACIÓN DEL GRÁFICO 2 (THETA)
    ax2.set_xlabel("Número de Tema", fontsize=15, fontweight='bold', color='#2c3e50')
    ax2.set_ylabel(f"Theta Escalado (×{theta_escalado_factor:.1f})", fontsize=15, fontweight='bold', color='#2c3e50')
    ax2.set_title("📈 Comparación Suavizada: Rendimiento (Theta) Escalado por Configuración (Temas 1-45)",
                  fontsize=17, fontweight='bold', pad=25, color='#2c3e50')

    # Configurar ejes para ambos gráficos
    for ax in [ax1, ax2]:
        ax.set_xlim(0.5, 45.5)
        ax.set_ylim(bottom=0)  # Empezar desde 0

        # Etiquetas del eje X cada 5 temas
        tick_positions = list(range(5, 46, 5))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(i) for i in tick_positions], fontsize=11)

        # Grid elegante
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, color='#bdc3c7')
        ax.set_axisbelow(True)

        # Líneas de referencia verticales cada 10 temas
        for x in range(10, 46, 10):
            ax.axvline(x=x, color='#ecf0f1', linewidth=1.5, alpha=0.6, zorder=0)

    # LEYENDAS mejoradas
    legend1 = ax1.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True,
                         shadow=True, framealpha=0.95, ncol=2 if len(configuraciones_guardadas) > 4 else 1)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_edgecolor('#cccccc')

    legend2 = ax2.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True,
                         shadow=True, framealpha=0.95, ncol=2 if len(configuraciones_guardadas) > 4 else 1)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_edgecolor('#cccccc')

    # Quitar spines superiores y derechos
    sns.despine()

    # INFORMACIÓN ESTADÍSTICA Y RESUMEN
    total_configs = len(configuraciones_guardadas)
    total_temas_con_datos = sum(1 for datos in configuraciones_completas.values()
                                for h in datos['horas'] if h > 0) // total_configs if total_configs > 0 else 0

    # Estadísticas de suavizado
    ax1.text(0.02, 0.98,
             f'🔄 Configuraciones: {total_configs}\n📚 Temas totales: 45\n📊 Con datos: {total_temas_con_datos}\n🎨 Suavizado: Spline Cúbico',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.7", facecolor='lightblue',
                       alpha=0.9, edgecolor='navy', linewidth=1.5),
             verticalalignment='top')

    ax2.text(0.02, 0.98,
             f'🎯 Factor escalado: {theta_escalado_factor:.2f}\n📊 Rango completo: 1-45\n🔍 Estilo: Líneas punteadas\n✨ Áreas sombreadas activas',
             transform=ax2.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.7", facecolor='lightgreen',
                       alpha=0.9, edgecolor='darkgreen', linewidth=1.5),
             verticalalignment='top')

    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08)

    plt.show()
    logger.info(f"📊 Gráfico suavizado mostrado - {len(configuraciones_guardadas)} configuraciones, 45 temas")


# Función de conveniencia para usar la versión suavizada por defecto
def plot_comparacion_configuraciones():
    """
    Muestra las configuraciones con el nuevo estilo suavizado.
    """
    plot_comparacion_configuraciones_suavizado()


def plot_configuracion_individual(resultado: Dict[str, int], omega: Dict[int, float], theta: Dict[int, float],
                                  nombre_config: str = "Configuración"):
    """
    Muestra una configuración individual con horas, importancia y theta escalado.
    """
    if not resultado:
        logger.warning("No hay resultados que graficar.")
        return

    # Preparar datos
    temas = sorted([int(k) for k in resultado.keys()])
    horas = [resultado[str(t)] for t in temas]
    importancias = [omega.get(t, 0) for t in temas]
    thetas = [theta.get(t, 0.1) for t in temas]

    # Escalar theta para comparación visual
    max_horas = max(horas) if horas else 1
    max_theta = max(thetas) if thetas else 1
    theta_escalado_factor = max_horas / max_theta
    thetas_escalados = [t * theta_escalado_factor for t in thetas]

    # Configurar estilo
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 12))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('white')

    # SUAVIZADO con spline para horas
    if len(temas) > 3:
        x_smooth = np.linspace(min(temas), max(temas), 300)
        try:
            # Horas
            spl_horas = make_interp_spline(temas, horas, k=min(3, len(temas) - 1))
            y_smooth_horas = spl_horas(x_smooth)
            y_smooth_horas = np.maximum(y_smooth_horas, 0)

            # Theta escalado
            spl_theta = make_interp_spline(temas, thetas_escalados, k=min(3, len(temas) - 1))
            y_smooth_theta = spl_theta(x_smooth)
            y_smooth_theta = np.maximum(y_smooth_theta, 0)

            # ÁREAS bajo las curvas
            ax.fill_between(x_smooth, 0, y_smooth_horas, color='#3498db', alpha=0.3,
                            label='Área Horas Asignadas', zorder=1)
            ax.fill_between(x_smooth, 0, y_smooth_theta, color='#e74c3c', alpha=0.2,
                            label='Área Theta Escalado', zorder=1)

            # LÍNEAS SUAVIZADAS
            ax.plot(x_smooth, y_smooth_horas, '-', color='#2980b9', linewidth=4,
                    alpha=0.9, label='Horas Asignadas', zorder=3)
            ax.plot(x_smooth, y_smooth_theta, '--', color='#c0392b', linewidth=3,
                    alpha=0.9, label=f'Theta Escalado (x{theta_escalado_factor:.1f})', zorder=3)

        except:
            # Fallback
            ax.plot(temas, horas, '-', color='#2980b9', linewidth=3, label='Horas Asignadas')
            ax.plot(temas, thetas_escalados, '--', color='#c0392b', linewidth=3,
                    label=f'Theta Escalado (x{theta_escalado_factor:.1f})')

    # PUNTOS con colores por importancia para horas
    norm = plt.Normalize(min(importancias), max(importancias))

    # Sombra de puntos
    ax.scatter(temas, horas, c='black', s=140, alpha=0.25, zorder=4)
    ax.scatter(temas, thetas_escalados, c='black', s=100, alpha=0.25, zorder=4, marker='s')

    # Puntos principales
    scatter_horas = ax.scatter(temas, horas, c=importancias, cmap='viridis',
                               s=120, zorder=5, edgecolors='white', linewidth=2.5,
                               alpha=0.95, norm=norm, label='Horas (por importancia)')

    scatter_theta = ax.scatter(temas, thetas_escalados, c='red', s=80, zorder=5,
                               edgecolors='white', linewidth=2, alpha=0.8, marker='s',
                               label='Theta Escalado')

    # ETIQUETAS de horas
    for tema, hora, theta_esc in zip(temas, horas, thetas_escalados):
        if hora > 0:
            ax.annotate(f'{hora}h', (tema, hora), textcoords="offset points",
                        xytext=(0, 18), ha='center', va='bottom', fontsize=10,
                        fontweight='bold', color='#2c3e50',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                                  alpha=0.8, edgecolor='#2980b9', linewidth=1))

        # Etiquetas para theta (solo cada 5 temas para no saturar)
        if tema % 5 == 0:
            theta_original = theta.get(tema, 0.1)
            ax.annotate(f'{theta_original:.2f}', (tema, theta_esc), textcoords="offset points",
                        xytext=(0, -20), ha='center', va='top', fontsize=9,
                        fontweight='bold', color='#c0392b',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral',
                                  alpha=0.8, edgecolor='#c0392b', linewidth=1))

    # CONFIGURACIÓN
    ax.set_xlabel("Número de Tema", fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_ylabel("Horas / Theta Escalado", fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_title(f"📚 Análisis Comparativo: {nombre_config} (Horas vs Theta)",
                 fontsize=18, fontweight='bold', pad=25, color='#2c3e50')

    # Límites
    ax.set_xlim(min(temas) - 2, max(temas) + 2)
    max_y = max(max(horas) if horas else 1, max(thetas_escalados) if thetas_escalados else 1)
    ax.set_ylim(-max_y * 0.05, max_y * 1.2)

    # Eje X
    step = max(1, len(temas) // 15)
    ax.set_xticks(temas[::step])
    ax.set_xticklabels([str(t) for t in temas[::step]], fontsize=12)

    # COLORBAR para importancia
    cbar = plt.colorbar(scatter_horas, ax=ax, shrink=0.8, aspect=25, pad=0.01)
    cbar.set_label('Importancia del Tema', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    # LEYENDA
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True,
                       shadow=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')

    # Grid y despine
    sns.despine()
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    # ESTADÍSTICAS
    total_horas = sum(horas)
    temas_activos = sum(1 for h in horas if h > 0)
    promedio_horas = total_horas / temas_activos if temas_activos > 0 else 0
    promedio_theta = np.mean(thetas)
    correlacion = np.corrcoef(horas, thetas)[0, 1] if len(horas) > 1 else 0

    stats_text = (f'📊 ESTADÍSTICAS:\n'
                  f'• Total horas: {total_horas}\n'
                  f'• Temas activos: {temas_activos}/{len(temas)}\n'
                  f'• Promedio horas: {promedio_horas:.1f}h\n'
                  f'• Promedio theta: {promedio_theta:.3f}\n'
                  f'• Correlación H-Θ: {correlacion:.3f}')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            fontweight='bold', bbox=dict(boxstyle="round,pad=0.6",
                                         facecolor='lightyellow', alpha=0.85,
                                         edgecolor='orange'),
            verticalalignment='top')

    plt.tight_layout()
    plt.show()
    logger.info(f"📊 Gráfico individual con theta mostrado: {nombre_config}")


def limpiar_configuraciones():
    """
    Limpia las configuraciones guardadas para empezar una nueva comparación.
    """
    global configuraciones_guardadas
    configuraciones_guardadas = []
    logger.info("🧹 Configuraciones guardadas limpiadas")


def asignar_horas_estudio_optimizado(
        # --- Parámetros Principales ---
        H_total: int,
        meses_estudio: List[int],
        vuelta_estudio: int,

        # --- Configuración General ---
        N: int,
        h_max_factor: float,
        alpha_factor: float,

        # --- Proporciones por Vuelta ---
        prop_b1: float,
        prop_b2: float,
        prop_b3: float,
        mu: float,

        # --- Penalizaciones ---
        lambda_penalty1: float,
        lambda_penalty2: float,
        lambda_penalty3: float,

        # --- Datos de Temas ---
        topic_blocks: Dict[int, int],
        omega: Dict[int, float],
        theta: Dict[int, float] = None  # NUEVO: Rendimiento por tema
) -> Dict[str, int]:
    """
    Calcula la asignación óptima de horas de estudio por tema incluyendo theta.
    """
    # Inicializar theta si no se proporciona
    if theta is None:
        theta = {i: 0.1 for i in range(1, N + 1)}
        logger.info("🎯 Theta inicializado a 0.1 para todos los temas")

    logger.info(f"🚀 Iniciando optimización - H_total: {H_total}, N: {N}, Vuelta: {vuelta_estudio}")

    try:
        # --- Validaciones iniciales ---
        if H_total <= 0 or not omega or not topic_blocks:
            raise ValueError("Parámetros de entrada inválidos")

        # Verificar proporciones
        suma_prop = prop_b1 + prop_b2 + prop_b3
        logger.info(f"📊 Proporciones - B1: {prop_b1}, B2: {prop_b2}, B3: {prop_b3}, Suma: {suma_prop:.3f}")
        if abs(suma_prop - 1.0) > 0.01:
            logger.warning(f"⚠️  Proporciones no suman 1.0: {suma_prop}")

        # --- Clasificar temas por bloque ---
        bloque_1 = [i - 1 for i, block in topic_blocks.items() if block == 1]
        bloque_2 = [i - 1 for i, block in topic_blocks.items() if block == 2]
        bloque_3 = [i - 1 for i, block in topic_blocks.items() if block == 3]

        logger.info(f"🏗️  Bloques - B1: {len(bloque_1)}, B2: {len(bloque_2)}, B3: {len(bloque_3)}")

        # ============================================
        # ANÁLISIS DE IMPORTANCIA Y THETA
        # ============================================
        importancias = list(omega.values())
        thetas_vals = list(theta.values())

        importancia_max = max(importancias) if importancias else 1
        importancia_min = min(importancias) if importancias else 0
        theta_max = max(thetas_vals) if thetas_vals else 1
        theta_min = min(thetas_vals) if thetas_vals else 0

        logger.info(f"📈 Importancias - Min: {importancia_min:.4f}, Max: {importancia_max:.4f}")
        logger.info(f"🎯 Theta - Min: {theta_min:.4f}, Max: {theta_max:.4f}")

        # Comprimir importancias para más equilibrio
        rango_original = importancia_max - importancia_min
        omega_comprimido = {}

        for tema, imp in omega.items():
            if rango_original > 0:
                imp_normalizada = (imp - importancia_min) / rango_original
                imp_comprimida = 0.6 + 0.4 * imp_normalizada  # Mapear a [0.6, 1.0]
                omega_comprimido[tema] = imp_comprimida * importancia_max
            else:
                omega_comprimido[tema] = imp

        logger.info(
            f"📉 Importancias comprimidas - Min: {min(omega_comprimido.values()):.4f}, Max: {max(omega_comprimido.values()):.4f}")

        # Identificar temas críticos
        umbral_critico = np.percentile(list(omega_comprimido.values()), 80)
        temas_criticos = [i for i, imp in omega_comprimido.items() if imp >= umbral_critico]
        logger.info(f"⭐ Temas críticos (>= {umbral_critico:.2f}): {temas_criticos}")

        # ============================================
        # ESTRATEGIA ADAPTATIVA CON THETA
        # ============================================
        horas_promedio_por_tema = H_total / N
        logger.info(f"⏱️  Horas promedio por tema: {horas_promedio_por_tema:.2f}")

        # Determinar estrategia
        if horas_promedio_por_tema >= 2.0:
            modo = "NORMAL"
            min_horas_base = max(1, int(H_total * alpha_factor))
            factor_importancia = 2.0
            garantizar_criticos = False
        elif horas_promedio_por_tema >= 1.0:
            modo = "AJUSTADO"
            min_horas_base = 0
            factor_importancia = 3.0
            garantizar_criticos = True
        else:
            modo = "CRITICO"
            min_horas_base = 0
            factor_importancia = 5.0
            garantizar_criticos = True

        logger.info(f"🎯 Modo: {modo}, Min_horas_base: {min_horas_base}, Garantizar_críticos: {garantizar_criticos}")

        # ============================================
        # CÁLCULO DE COEFICIENTES CON THETA
        # ============================================
        a = {}
        alpha = {}
        for i in range(1, N + 1):
            # Usar importancias comprimidas
            importancia = omega_comprimido.get(i, 1)
            theta_val = max(0.001, theta.get(i, 0.1))  # Evitar división por cero

            # Amplificar importancia según el modo
            if modo == "CRITICO" and rango_original > 0:
                importancia_normalizada = (importancia - importancia_min) / rango_original
                importancia_ajustada = importancia * (1 + factor_importancia * importancia_normalizada ** 2)
            else:
                importancia_ajustada = importancia ** (factor_importancia / 2)

            # INCORPORAR THETA: Mayor theta (mejor rendimiento) = menos horas necesarias
            a[i] = importancia_ajustada / (mu * theta_val)
            alpha[i] = max(1, min_horas_base) if garantizar_criticos and i in temas_criticos else min_horas_base

        logger.info(f"🧮 Coeficientes calculados con theta. Rango a: [{min(a.values()):.3f}, {max(a.values()):.3f}]")

        # ============================================
        # CONFIGURACIÓN DE RESTRICCIONES
        # ============================================
        horas_objetivo_b1 = prop_b1 * H_total
        horas_objetivo_b2 = prop_b2 * H_total
        horas_objetivo_b3 = prop_b3 * H_total

        H_max = max(H_total * h_max_factor, 1)
        a_vec = np.array([a.get(i, 0) for i in range(1, N + 1)])
        alpha_vec = np.array([alpha.get(i, 0) for i in range(1, N + 1)])

        tolerancia = 0.15 * H_total if modo == "CRITICO" else 0.1 * H_total

        # Verificación de factibilidad
        suma_minimas = np.sum(alpha_vec)
        logger.info(f"🧮 Horas mínimas totales: {suma_minimas:.2f} / {H_total}")
        if suma_minimas > H_total:
            logger.error(f"❌ INFACTIBLE: Suma de mínimas ({suma_minimas}) > H_total ({H_total})")
            return asignacion_por_prioridad_simple(N, H_total, omega_comprimido, temas_criticos, theta)

        # Verificar cada bloque
        for bloque, horas_obj, nombre in [
            (bloque_1, horas_objetivo_b1, "B1"),
            (bloque_2, horas_objetivo_b2, "B2"),
            (bloque_3, horas_objetivo_b3, "B3")
        ]:
            if bloque:
                min_req_bloque = sum(alpha_vec[i] for i in bloque)
                max_permitido = horas_obj + tolerancia
                logger.info(
                    f"🔍 {nombre}: min_req={min_req_bloque:.1f}, obj={horas_obj:.1f}, max_perm={max_permitido:.1f}")

                if min_req_bloque > max_permitido:
                    logger.error(f"❌ INFACTIBLE {nombre}: {min_req_bloque:.2f} > {max_permitido:.2f}")
                    return asignacion_por_prioridad_simple(N, H_total, omega_comprimido, temas_criticos, theta)

        # ============================================
        # VARIABLES Y RESTRICCIONES CVXPY
        # ============================================
        lambda_penalty_vec = np.zeros(N)
        lambda_penalty_vec[bloque_1] = lambda_penalty1
        lambda_penalty_vec[bloque_2] = lambda_penalty2
        lambda_penalty_vec[bloque_3] = lambda_penalty3

        h = cp.Variable(N, nonneg=True)
        p = cp.Variable(N, nonneg=True)
        slack_b1 = cp.Variable(nonneg=True)
        slack_b2 = cp.Variable(nonneg=True)
        slack_b3 = cp.Variable(nonneg=True)

        penalty_slack = 50 if modo == "CRITICO" else (200 if modo == "AJUSTADO" else 1000)

        # Función objetivo logarítmica con theta para evitar concentraciones extremas
        objective = cp.Maximize(
            cp.sum(cp.multiply(a_vec, cp.log(h + 1))) -
            cp.sum(cp.multiply(lambda_penalty_vec, cp.square(p))) -
            penalty_slack * (slack_b1 + slack_b2 + slack_b3)
        )

        logger.info(f"🎯 Función objetivo: logarítmica con theta incorporado")

        # Restricciones base
        constraints = [cp.sum(h) == H_total, p >= h - H_max]

        # Restricción de equilibrio para evitar concentraciones extremas
        promedio_horas = H_total / N
        max_desviacion_permitida = promedio_horas * 0.8
        constraints.append(h <= promedio_horas + max_desviacion_permitida)
        constraints.append(h >= promedio_horas - max_desviacion_permitida)
        logger.info(
            f"🔧 Restricción de equilibrio: h ∈ [{promedio_horas - max_desviacion_permitida:.1f}, {promedio_horas + max_desviacion_permitida:.1f}]")

        # Restricciones de horas mínimas
        if not garantizar_criticos or horas_promedio_por_tema >= 0.5:
            constraints.append(h >= alpha_vec)
            logger.info(f"🔧 Horas mínimas aplicadas a todos los temas")
        else:
            count_criticos = 0
            for i in range(N):
                if (i + 1) in temas_criticos:
                    constraints.append(h[i] >= alpha_vec[i])
                    count_criticos += 1
            logger.info(f"🔧 Horas mínimas solo para {count_criticos} temas críticos")

        # Restricciones de proporción por bloque con slack
        if bloque_1:
            constraints.extend([
                cp.sum(h[bloque_1]) <= horas_objetivo_b1 + slack_b1,
                cp.sum(h[bloque_1]) >= horas_objetivo_b1 - slack_b1,
                slack_b1 <= tolerancia
            ])
            logger.info(f"🔧 B1: [{horas_objetivo_b1 - tolerancia:.1f}, {horas_objetivo_b1 + tolerancia:.1f}]")

        if bloque_2:
            constraints.extend([
                cp.sum(h[bloque_2]) <= horas_objetivo_b2 + slack_b2,
                cp.sum(h[bloque_2]) >= horas_objetivo_b2 - slack_b2,
                slack_b2 <= tolerancia
            ])
            logger.info(f"🔧 B2: [{horas_objetivo_b2 - tolerancia:.1f}, {horas_objetivo_b2 + tolerancia:.1f}]")

        if bloque_3:
            constraints.extend([
                cp.sum(h[bloque_3]) <= horas_objetivo_b3 + slack_b3,
                cp.sum(h[bloque_3]) >= horas_objetivo_b3 - slack_b3,
                slack_b3 <= tolerancia
            ])
            logger.info(f"🔧 B3: [{horas_objetivo_b3 - tolerancia:.1f}, {horas_objetivo_b3 + tolerancia:.1f}]")

        logger.info(f"📏 Total restricciones: {len(constraints)}")

        # ============================================
        # RESOLVER EL PROBLEMA
        # ============================================
        problem = cp.Problem(objective, constraints)
        solved = False
        solver_usado = "Ninguno"

        # Intentar con diferentes solvers
        solvers_disponibles = []
        if hasattr(cp, 'ECOS'):
            solvers_disponibles.append(('ECOS', cp.ECOS))
        if hasattr(cp, 'SCS'):
            solvers_disponibles.append(('SCS', cp.SCS))
        if hasattr(cp, 'OSQP'):
            solvers_disponibles.append(('OSQP', cp.OSQP))
        if hasattr(cp, 'CLARABEL'):
            solvers_disponibles.append(('CLARABEL', cp.CLARABEL))

        logger.info(f"🔧 Solvers disponibles: {[nombre for nombre, _ in solvers_disponibles]}")

        for solver_name, solver in solvers_disponibles:
            try:
                logger.info(f"🔧 Probando solver: {solver_name}")
                problem.solve(solver=solver, verbose=False)

                logger.info(f"   Status: {problem.status}")
                if problem.value is not None:
                    logger.info(f"   Valor óptimo: {problem.value:.4f}")

                if problem.status not in ["infeasible", "unbounded"]:
                    solver_usado = solver_name
                    solved = True
                    logger.info(f"✅ Resuelto con {solver_name}")
                    break

            except Exception as e:
                logger.error(f"❌ Error con {solver_name}: {e}")
                continue

        if not solved or h.value is None:
            logger.error("💥 TODOS LOS SOLVERS FALLARON")
            return asignacion_por_prioridad_simple(N, H_total, omega_comprimido, temas_criticos, theta)

        # ============================================
        # PROCESAR SOLUCIÓN
        # ============================================
        horas_brutas = h.value
        logger.info(f"✅ Solución obtenida - Suma bruta: {np.sum(horas_brutas):.4f}")
        logger.info(f"   Rango: [{np.min(horas_brutas):.2f}, {np.max(horas_brutas):.2f}]")

        # Análisis de equilibrio
        desviacion_estandar = np.std(horas_brutas)
        coeficiente_variacion = desviacion_estandar / np.mean(horas_brutas) if np.mean(horas_brutas) > 0 else 0
        logger.info(
            f"📊 Análisis de equilibrio - Desv.estándar: {desviacion_estandar:.2f}, Coef.variación: {coeficiente_variacion:.3f}")

        # Análisis de correlación horas-theta
        horas_por_tema = {i + 1: horas_brutas[i] for i in range(N)}
        correlacion_horas_theta = np.corrcoef(
            [horas_por_tema.get(i, 0) for i in range(1, N + 1)],
            [theta.get(i, 0.1) for i in range(1, N + 1)]
        )[0, 1] if N > 1 else 0

        logger.info(f"🎯 Correlación Horas-Theta: {correlacion_horas_theta:.3f} (negativa esperada)")

        # Redondeo y ajuste
        horas_asignadas = np.round(horas_brutas).astype(int)

        # Ajuste final para sumar exactamente H_total
        diferencia = H_total - np.sum(horas_asignadas)
        if diferencia != 0:
            logger.info(f"🔧 Ajustando diferencia: {diferencia}")
            ajustar_diferencia_final(horas_asignadas, diferencia, a_vec, temas_criticos)

        resultado = {str(i + 1): int(h) for i, h in enumerate(horas_asignadas)}

        # ============================================
        # REPORTE FINAL
        # ============================================
        temas_cubiertos = sum(1 for h in resultado.values() if h > 0)
        temas_criticos_cubiertos = sum(1 for t in temas_criticos if resultado[str(t)] > 0)

        logger.info(f"📊 Resultado: {temas_cubiertos}/{N} temas cubiertos")
        logger.info(f"⭐ Temas críticos cubiertos: {temas_criticos_cubiertos}/{len(temas_criticos)}")

        # Verificar resultado por bloques
        for bloque, horas_obj, nombre in [
            (bloque_1, horas_objetivo_b1, "B1"),
            (bloque_2, horas_objetivo_b2, "B2"),
            (bloque_3, horas_objetivo_b3, "B3")
        ]:
            if bloque:
                horas_finales = sum(resultado[str(i + 1)] for i in bloque)
                desv = abs(horas_finales - horas_obj)
                logger.info(f"   {nombre}: {horas_finales}h (obj: {horas_obj:.1f}h, desv: {desv:.1f}h)")

        # Top 10 temas con más horas vs su theta
        top_temas = sorted([(k, v) for k, v in resultado.items() if v > 0],
                           key=lambda x: x[1], reverse=True)[:10]
        logger.info("📈 Top 10 temas con más horas:")
        for tema, horas in top_temas:
            imp = omega_comprimido.get(int(tema), 0)
            theta_val = theta.get(int(tema), 0.1)
            eficiencia = horas / theta_val if theta_val > 0 else 0
            logger.info(f"   Tema {tema}: {horas}h (imp: {imp:.2f}, θ: {theta_val:.3f}, efic: {eficiencia:.1f})")

        # Advertir si algún tema crítico no fue cubierto
        criticos_sin_cubrir = [t for t in temas_criticos if resultado[str(t)] == 0]
        if criticos_sin_cubrir:
            logger.warning(f"⚠️  Temas críticos SIN CUBRIR: {criticos_sin_cubrir}")

        # Mostrar gráfico con theta
        plot_and_save_results(resultado, omega, theta, f"Optimizado_V{vuelta_estudio}")

        return resultado

    except Exception as e:
        logger.error(f"💥 Error crítico: {e}", exc_info=True)
        raise


def asignacion_por_prioridad_simple(N, H_total, omega, temas_criticos, theta=None):
    """
    Asignación de emergencia simple basada en importancia y theta.
    """
    logger.warning("🚨 Asignación de emergencia activada")

    if theta is None:
        theta = {i: 0.1 for i in range(1, N + 1)}

    prioridades = []
    for i in range(1, N + 1):
        importancia = omega.get(i, 1)
        rendimiento = max(0.001, theta.get(i, 0.1))
        # Mayor importancia y menor rendimiento = mayor prioridad
        score = importancia / rendimiento
        if i in temas_criticos:
            score *= 2
        prioridades.append((i, score))

    prioridades.sort(key=lambda x: x[1], reverse=True)

    resultado = {str(i): 0 for i in range(1, N + 1)}
    horas_restantes = H_total

    # Distribuir horas una por una siguiendo el orden de prioridad
    while horas_restantes > 0:
        for tema, score in prioridades:
            if horas_restantes <= 0:
                break
            resultado[str(tema)] += 1
            horas_restantes -= 1

    logger.info(f"✅ Emergencia completada: {sum(1 for h in resultado.values() if h > 0)}/{N} temas")
    return resultado


def ajustar_diferencia_final(horas_asignadas, diferencia, utilidades, temas_criticos):
    """
    Ajusta las horas finales para que la suma sea exactamente H_total.
    """
    N = len(horas_asignadas)
    indices_ordenados = np.argsort(utilidades)

    if diferencia > 0:  # Faltan horas
        logger.debug(f"   ➕ Añadiendo {diferencia} horas a temas de mayor utilidad")
        for i in np.flip(indices_ordenados):
            if diferencia <= 0:
                break
            horas_asignadas[i] += 1
            diferencia -= 1

    elif diferencia < 0:  # Sobran horas
        logger.debug(f"   ➖ Quitando {abs(diferencia)} horas de temas de menor utilidad")
        for i in indices_ordenados:
            if diferencia >= 0:
                break
            es_critico = (i + 1) in temas_criticos
            limite_inferior = 1 if es_critico else 0
            if horas_asignadas[i] > limite_inferior:
                horas_asignadas[i] -= 1
                diferencia += 1

    logger.debug(f"   ✅ Ajuste completado. Diferencia final: {diferencia}")


# === EJEMPLO DE USO ===
def ejemplo_uso():
    """
    Ejemplo de cómo usar el algoritmo optimizado con theta.
    """
    # Datos de ejemplo
    N = 45
    H_total = 120

    # Configuración general
    h_max_factor = 0.15
    alpha_factor = 0.02

    # Proporciones por vuelta
    prop_b1 = 0.4
    prop_b2 = 0.35
    prop_b3 = 0.25
    mu = 1.0

    # Penalizaciones
    lambda_penalty1 = 0.1
    lambda_penalty2 = 0.1
    lambda_penalty3 = 0.1

    # Datos de temas (ejemplo)
    topic_blocks = {i: ((i - 1) % 3) + 1 for i in range(1, N + 1)}  # Distribuir en 3 bloques
    omega = {i: np.random.uniform(0.5, 1.0) for i in range(1, N + 1)}  # Importancias aleatorias
    theta = {i: np.random.uniform(0.05, 0.2) for i in range(1, N + 1)}  # Rendimientos aleatorios

    # Ejecutar optimización
    resultado = asignar_horas_estudio_optimizado(
        H_total=H_total,
        meses_estudio=[1, 2, 3],
        vuelta_estudio=1,
        N=N,
        h_max_factor=h_max_factor,
        alpha_factor=alpha_factor,
        prop_b1=prop_b1,
        prop_b2=prop_b2,
        prop_b3=prop_b3,
        mu=mu,
        lambda_penalty1=lambda_penalty1,
        lambda_penalty2=lambda_penalty2,
        lambda_penalty3=lambda_penalty3,
        topic_blocks=topic_blocks,
        omega=omega,
        theta=theta
    )

    return resultado


# Ejemplo con diferentes configuraciones de theta para comparar
def comparar_diferentes_thetas():
    """
    Compara el algoritmo con diferentes configuraciones de theta.
    """
    limpiar_configuraciones()

    # Configuración base
    N = 30
    H_total = 90
    topic_blocks = {i: ((i - 1) % 3) + 1 for i in range(1, N + 1)}
    omega = {i: np.random.uniform(0.3, 1.0) for i in range(1, N + 1)}

    # Diferentes configuraciones de theta
    configuraciones_theta = {
        "Theta Uniforme (0.1)": {i: 0.1 for i in range(1, N + 1)},
        "Theta Variable": {i: np.random.uniform(0.05, 0.2) for i in range(1, N + 1)},
        "Theta Alto": {i: np.random.uniform(0.15, 0.3) for i in range(1, N + 1)},
    }

    # Ejecutar cada configuración
    for nombre, theta_config in configuraciones_theta.items():
        logger.info(f"\n🔄 Ejecutando configuración: {nombre}")

        resultado = asignar_horas_estudio_optimizado(
            H_total=H_total,
            meses_estudio=[1, 2, 3],
            vuelta_estudio=1,
            N=N,
            h_max_factor=0.15,
            alpha_factor=0.02,
            prop_b1=0.4,
            prop_b2=0.35,
            prop_b3=0.25,
            mu=1.0,
            lambda_penalty1=0.1,
            lambda_penalty2=0.1,
            lambda_penalty3=0.1,
            topic_blocks=topic_blocks,
            omega=omega,
            theta=theta_config
        )

        # Guardar configuración para comparación (sin mostrar aún)
        plot_and_save_results(resultado, omega, theta_config, nombre, mostrar_ahora=False)

    # Mostrar comparación final
    plot_comparacion_configuraciones()


if __name__ == "__main__":
    # Ejemplo básico
    print("🚀 Ejecutando ejemplo básico...")
    resultado_ejemplo = ejemplo_uso()

    # Comparación de configuraciones
    print("\n🔄 Comparando diferentes configuraciones de theta...")
    comparar_diferentes_thetas()