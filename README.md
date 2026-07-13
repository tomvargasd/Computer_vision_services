# CVVision Demo — Detección de Personas, Armas y Acciones

## Descripción

Este proyecto es una demo integral de visión por computadora para la detección en tiempo real de personas, armas y acciones sospechosas/violentas en video. Utiliza modelos de deep learning (YOLOv8 pose y custom armas/personas), OpenCV y un backend Flask con una interfaz web moderna.

## Características principales

- **Detección de personas**: Conteo, permanencia, heatmap de movimiento.
- **Detección de armas**: Identificación de armas, captura automática de rostro, clasificación de tipo de arma.
- **Detección de acciones**: Detección de violencia, robo/amenaza, actividad sospechosa (agachado, rastreo, movimientos inusuales).
- **Asignación de ID persistente**: Seguimiento de personas a través de frames.
- **Captura automática**: Alerta y captura de rostro/cuerpo ante eventos.
- **Galería de capturas**: Visualización y lightbox de imágenes de alerta.
- **Streaming MJPEG**: Visualización en tiempo real.
- **Configuración persistente**: SQLite WAL para settings y metadatos.
- **UI moderna**: Flask + Jinja2, CSS custom, lightbox, toggles y leyendas.

## Estructura del proyecto

```
app.py                     # Punto de entrada principal (ejecutable)
database.py                # Inicialización de la base de datos local SQLite
bytetrack_armas.yaml       # Configuración para tracking de armas con ByteTrack
requirements.txt           # Dependencias del sistema
.gitignore                 # Exclusiones de control de versiones (modelos, videos y db local)
src/                       # Código fuente de la aplicación
  ├── app.py               # Backend Flask y lógica de endpoints
  ├── config.py            # Configuraciones del sistema y variables de entorno
  ├── database.py          # Gestión de base de datos SQLite y persistencia de alertas
  ├── utils.py             # Utilidades de ayuda generales
  ├── routes/              # Módulos de enrutamiento
  └── modules/             # Procesamiento y lógica de Visión por Computadora (YOLOv8)
        ├── base.py            # Clase base para todos los módulos de visión
        ├── personas.py        # Módulo de tracking, conteo y heatmap de personas
        ├── armas.py           # Módulo de detección de armas y captura de rostros
        ├── acciones.py        # Módulo de pose y detección de acciones sospechosas/violencia
        ├── troncos.py         # Conteo y tracking de troncos de madera
        ├── pallets.py         # Conteo y tracking de pallets en zonas específicas
        ├── cajas.py           # Conteo de cajas cruzando líneas virtuales
        ├── reglamento.py      # Detección de EPP/reglamentos (ej. uso de botas)
        ├── carga_descarga.py  # Detección de procesos de carga y descarga
        ├── epp.py             # Detección de Elementos de Protección Personal
        ├── smoke.py           # Detección de fuego y humo
        └── vehiculos.py       # Detección y conteo de vehículos
static/                    # Archivos estáticos
  ├── css/                 # Estilos CSS de la interfaz visual
  ├── js/                  # Scripts interactivos JavaScript
  └── uploads/             # Directorio de subidas (capturas de alertas, modelos .pt y videos .mp4)
templates/                 # Vistas HTML con Jinja2 (Dashboard, alertas, configuraciones)
```

## Requisitos

- Python 3.9+
- pip
- macOS, Linux o Windows

### Dependencias principales
- Flask 3.1.0
- Flask-CORS
- OpenCV (cv2)
- Ultralytics YOLOv8
- numpy
- sqlite3

Instala dependencias con:
```bash
pip install -r requirements.txt
```

## Uso rápido

1. Clona el repositorio y entra al directorio:
   ```bash
   git clone <repo_url>
   cd <repo>
   ```
2. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # o venv\Scripts\activate en Windows
   ```
3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. **Modelos y videos**: Asegúrate de descargar y colocar tus modelos YOLO `.pt` en `static/uploads/models/` y los videos de prueba en `static/uploads/videos/` (estos directorios están excluidos del repositorio Git para mantener un tamaño ligero).
5. Ejecuta la app:
   ```bash
   python app.py
   ```
6. Abre tu navegador en [http://localhost:5001](http://localhost:5001) (o el puerto configurado).

## Despliegue con Docker y Aceleración GPU (CUDA)

El proyecto cuenta con soporte para Docker y Docker Compose, facilitando su ejecución tanto en entornos locales con CPU (como macOS o Windows) como en servidores con aceleración por GPU NVIDIA (Linux).

El script de despliegue (`deploy.sh`) autodetecta el sistema operativo, la presencia de GPUs NVIDIA y la versión de CUDA en el host para descargar automáticamente la versión de PyTorch adecuada dentro del contenedor.

### Requisitos previos para Linux con GPU (CUDA)

Para que el procesamiento se realice en la GPU de NVIDIA dentro del contenedor Docker en Linux, debes cumplir con los siguientes requisitos en el host:

1. **Controladores de NVIDIA**: Tener instalados los drivers oficiales y actualizados en el sistema operativo host.
2. **Docker y Docker Compose**: Tener instalado el motor Docker (versión 20.10+) y el plugin Docker Compose.
3. **NVIDIA Container Toolkit**: Es el componente crítico que permite a Docker interactuar con la GPU de la máquina host.
   - Instálalo siguiendo la guía oficial: [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   - Configura el runtime de NVIDIA para Docker y reinicia el servicio daemon:
     ```bash
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo systemctl restart docker
     ```

*Nota: Si estás en macOS o Windows/Linux sin GPU NVIDIA, el script detectará la ausencia de CUDA y desplegará de forma segura en modo **CPU** utilizando la librería optimizada para ello.*

### Despliegue rápido con Docker

1. **Modelos y videos**: Asegúrate de tener los modelos YOLO `.pt` en `static/uploads/models/` y los videos de prueba en `static/uploads/videos/`.
2. **Ejecutar script de despliegue**:
   ```bash
   ./deploy.sh
   ```
3. El script creará la base de datos `cvvision.db` vacía en el host (si no existe), creará las carpetas necesarias en `static/uploads`, compilará la imagen de Docker adecuada e iniciará el contenedor en segundo plano.
4. **Acceder a la aplicación**: Abre tu navegador en [http://localhost:5001](http://localhost:5001).

### Comandos útiles de Docker

* **Ver logs del contenedor**:
  ```bash
  docker compose logs -f
  ```
* **Detener la aplicación**:
  ```bash
  docker compose down
  ```
* **Reconstruir contenedores**:
  ```bash
  docker compose build --no-cache
  ```

## Configuración y módulos

- El sistema cuenta con múltiples módulos de visión artificial que se pueden activar o desactivar independientemente desde la UI:
  - **Detección de Personas**
  - **Detección de Armas**
  - **Detección de Acciones (Violencia/Robo/Actividad sospechosa)**
  - **Conteo de Troncos**
  - **Conteo de Pallets**
  - **Conteo de Cajas**
  - **Reglamento / EPP**
  - **Carga y Descarga**
  - **Elementos de Protección Personal (EPP)**
  - **Detección de Humo y Fuego**
  - **Detección de Vehículos**
- Puedes registrar videos o streams (cámaras RTSP) como fuentes de entrada para cada módulo.
- Los parámetros de confianza, zonas/líneas de conteo y precisión se ajustan en tiempo real desde la interfaz.
- Las capturas automáticas de alertas se almacenan localmente en `static/uploads/captures/`.

## Personalización

- Puedes cambiar los modelos YOLO colocando archivos `.pt` en `static/uploads/models/` y seleccionándolos desde la interfaz.
- Los umbrales y la lógica de detección de alertas de visión pueden modificarse directamente en los archivos correspondientes dentro de `src/modules/`.

## Notas técnicas

- El sistema usa SQLite en modo WAL para evitar bloqueos y mejorar concurrencia.
- El streaming de video es MJPEG (compatible con la mayoría de navegadores).
- El seguimiento de personas usa centroid matching y asignación de TID.
- La lógica de acciones es geométrica y configurable.

## Créditos

- Basado en Ultralytics YOLOv8, OpenCV y Flask.
- Demo desarrollada por [tu nombre o equipo].

## Licencia

MIT License.
