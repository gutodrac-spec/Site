import folium
import gradio as gr
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="site-mapa-gradio")


def gerar_mapa(local, lat, lon, zoom):
    if local:
        loc = geolocator.geocode(local)
        if loc is None:
            return f"<p>Local não encontrado: {local}</p>"
        lat, lon = loc.latitude, loc.longitude
        popup = loc.address
    else:
        popup = f"{lat}, {lon}"
    m = folium.Map(location=[lat, lon], zoom_start=int(zoom))
    folium.Marker([lat, lon], popup=popup).add_to(m)
    return m._repr_html_()


with gr.Blocks(title="Mapa") as demo:
    gr.Markdown("# Mapa\nDigite um endereço ou use latitude/longitude.")
    with gr.Row():
        local = gr.Textbox(label="Endereço / cidade", placeholder="Ex.: Avenida Paulista, São Paulo")
        lat = gr.Number(label="Latitude", value=-23.5505)
        lon = gr.Number(label="Longitude", value=-46.6333)
        zoom = gr.Slider(1, 18, value=12, step=1, label="Zoom")
    botao = gr.Button("Mostrar mapa")
    saida = gr.HTML(gerar_mapa("", -23.5505, -46.6333, 12))
    botao.click(gerar_mapa, [local, lat, lon, zoom], saida)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
