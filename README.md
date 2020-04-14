# UOC Tipologia de Datos


## Getting Started

La temática elegida para nuestro trabajo a sido la actual sobre el Coronavirus o COVID-19. Este virus ha entrado en la vida de todas las personas del planeta desde hace unos pocos meses, llevando su letalidad a parar las actividades normales de todo tipo (trabajos, rutinas…). En la actualidad, este es un tema recurrente y sobre el que se tiene información casi las 24 horas del día, debido a su notoriedad y magnitud.

En cuanto a las fuentes elegidas, se ha seleccionado la web oficial del gobierno de España y la web Worldometers como fuentes estadísticas. Ambas webs proporcionan datos actualizados del virus, la primera de España y la segunda a nivel mundial. De esta forma, se consideran fuentes fiables debido a que, en la primera, el gobierno es el que proporcionaría los datos; y la segunda debido a que, tras realizar un proceso de investigación, se ha visto que los datos que utilizan se corresponden con los de los gobiernos correspondientes.

Además, hemos utilizado Twitter como fuente textual, una red sociales en la que se puede recolectar datos de forma más sencilla y, sobre todo, inmediata, necesario para  establecer correlaciones temporales entre la evolución del virus y la opinión del mismo de la sociedad.

Esta plataforma permite él envió de mensajes en texto plano de corta longitud por parte de los usuarios, con un máximo de 280 caracteres. Estos mensajes, llamados tweets, se muestran en la página principal del usuario y pueden ser capturados a través de una API proporcionada por la propia red social.

### Files

En este repositorio se pueden encontrar los siguientes ficheros:
```
-Tweepy.py - Este fichero recoge el código que ejecuta el web scraping y recoge los tweets, así como la formación del dataset final.
-chromedriver.exe - Este ejecutable es necesario para la realización del web scraping en el código (debe estar en consonancia con la versión de Google Chrome que se posea).
-autenticate.py - Este archivo es un fichero auxiliar donde se referencian las credenciales para acceder a la API de Twitter y obtener los tweets.
-base_de_datos_covid.csv - Este fichero recogería el dataset final tras la ejecución del código mencionado.
-requirements.txt - Este archivo menciona los paquetes Python necesarios para ejecutar el código.
```

### Prerequisites

```
tweepy
bs4
selenium
time
pandas
```

### Installing
Para ejecutar este proyecto es necesario ejecutar el siguiente comando y añadir las credenciales de acceso a la api de twitter. 

```
python get-pip.py install -r requirements.txt
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/al118345/tipologia_uoc/edit/master/tags). 

## Authors

* Roberto Alexander Cerviño Cortinez
* Rubén Pérez Ibáñez

## License
Released Under CC BY-SA 4.0 License


