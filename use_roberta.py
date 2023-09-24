import logging
import time
from utils.use_llms_models import generate_summary_roberta

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info(f"New LLM")
logging.info("Start")

text = """En el siglo XVIII, en el corazón de la Revolución Industrial, las fábricas se alzaban como gigantes de 
acero y humo. Las chimeneas escupían columnas de humo negro que oscurecían el cielo. Los trabajadores, en su mayoría 
inmigrantes y niños, llenaban estas fábricas, realizando largas jornadas de trabajo en condiciones peligrosas. La 
maquinaria pesada y despiadada requería una atención constante, y los accidentes eran comunes.

Sin embargo, la Revolución Industrial también trajo consigo avances tecnológicos que transformaron la sociedad. Las 
máquinas de hilar y tejer automatizadas revolucionaron la industria textil, aumentando la producción y reduciendo los 
costos. El ferrocarril se expandió rápidamente, conectando ciudades y regiones de una manera que nunca antes se había 
visto. Los barcos de vapor conquistaron los océanos, acortando los viajes transatlánticos. La industrialización 
impulsó el crecimiento económico, aunque a menudo a expensas de la clase trabajadora.

El movimiento obrero surgió como respuesta a las condiciones laborales injustas. Los sindicatos se formaron para 
luchar por salarios más altos, jornadas laborales más cortas y condiciones de trabajo más seguras. Hubo protestas y 
huelgas que a menudo fueron reprimidas con violencia por parte de las autoridades y las empresas. A pesar de los 
desafíos, el movimiento obrero gradualmente logró mejoras significativas en las condiciones de trabajo.

La Revolución Industrial no solo transformó la economía y el trabajo, sino que también tuvo un impacto profundo en la 
sociedad en general. Cambió la forma en que las personas vivían y trabajaban, reconfigurando las ciudades y creando 
nuevas clases sociales. A medida que las fábricas se multiplicaban, la vida rural se desvanecía gradualmente y la 
urbanización se aceleraba. La Revolución Industrial, con todos sus avances y desafíos, dejó una huella indeleble en 
la historia de la humanidad."""

print(f"FULL TEXT:\n{text}\n")
print(f"SUMMARY:\n{generate_summary_roberta(text)}")
time.sleep(3)
logging.info("End")
