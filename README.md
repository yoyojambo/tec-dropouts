# Configuración para Colaborar en Notebooks (Jupyter)

Para evitar conflictos en GitHub, usamos una herramienta llamada pre-commit. Esta
herramienta limpia automáticamente los resultados de las celdas (gráficas, tablas,
ejecuciones) en segundo plano justo antes de hacer un commit, manteniendo el historial de
Git limpio sin alterar lo que ves en tu pantalla.

## Configuración Única (Solo se hace una vez)

Abre tu terminal en la carpeta raíz del proyecto. Instala las herramientas necesarias
ejecutando:

```bash
pip install pre-commit nbstripout
```

Activa el hook en tu Git local con este comando:

```bash
pre-commit install
```
