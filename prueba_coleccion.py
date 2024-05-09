import os
import logging
import datetime

log_dir = '/home/ladyotavo/nlp/proyecto/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Obtener la fecha y hora actual
current_time = datetime.datetime.now()

# Formatear la fecha y hora en el formato deseado
log_filename = current_time.strftime('procesamiento_%Y-%m-%d_%H-%M-%S.log')

# Configurar el registro con el nombre de archivo generado
logging.basicConfig(filename=os.path.join(log_dir, log_filename), level=logging.INFO, format='%(asctime)s - %(message)s')

# Ejemplo de registro
logging.info("Mensaje de prueba para mi log")
