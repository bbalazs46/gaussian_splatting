# gaussian_splatting

Az alkalmazás jelenleg tartalmaz egy egyszerű képmappa-alapú Gaussian Splatting előkészítő workflow-t is:

- `open_image_folder(folder_path)`: támogatott képek összegyűjtése egy mappából
- `create_gaussian_scene_file(folder_path)`: létrehoz egy `gaussian_scene.json` fájlt a mappában
- `evaluate_gaussian_scene_consistency(scene_data, folder_path=None)`: 0..1 közötti konzisztencia pontszámot ad
- `improve_gaussian_scene_consistency(scene_data, folder_path=None, step_size=0.5)`: javítja a jelenet pontszámát

A létrehozott JSON fájl az egyes képekhez kezdeti, durva kamera- és Gaussian-becslést tárol. A `C` billentyű ezek konzisztenciáját méri, az `I` billentyű pedig a képek statisztikái felé javítja a jelenetet.

A viewerben az `O` billentyűvel mappaválasztó ablak nyitható, amivel kijelölhető a használandó képmappa. A `C` billentyű kiértékeli a meglévő `gaussian_scene.json` konzisztenciáját, az `I` billentyű pedig javítja és elmenti a jelenetet.
