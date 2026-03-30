# gaussian_splatting

Az alkalmazás jelenleg tartalmaz egy egyszerű képmappa-alapú Gaussian Splatting előkészítő workflow-t is:

- `open_image_folder(folder_path)`: támogatott képek összegyűjtése egy mappából
- `create_gaussian_scene_file(folder_path)`: létrehoz egy `gaussian_scene.json` fájlt a mappában
- `evaluate_gaussian_scene_consistency(scene_data, folder_path=None)`: 0..1 közötti konzisztencia pontszámot ad
- `improve_gaussian_scene_consistency(scene_data, folder_path=None, step_size=0.5)`: javítja a jelenet pontszámát
- `randomize_gaussian_scene(scene_data, folder_path=None)`: randomizálja a meglévő Gaussian-foltokat
- `append_random_gaussian_scene(scene_data, folder_path=None)`: hozzáad egy új random Gaussian-foltot

A létrehozott JSON fájl az egyes képekhez kezdeti, durva kamera- és Gaussian-becslést tárol. A `C` billentyű ezek konzisztenciáját méri, az `I` billentyű pedig a képek statisztikái felé javítja a jelenetet. A `+/-` billentyűkkel a javítás lépésköze állítható, az `F` billentyűvel pedig folyamatos, lépésenkénti javítás kapcsolható be vagy ki. Az `N` billentyű egy új random Gaussian-foltot ad hozzá a jelenethez.

A viewerben az `O` billentyűvel mappaválasztó ablak nyitható, amivel kijelölhető a használandó képmappa. Ha a mappában van `gaussian_scene.json`, akkor a viewer ezt a Gaussian-jelenetet rendereli. A `C` billentyű ennek a jelenetnek a konzisztenciáját értékeli ki, az `I` billentyű a beállított sebességgel javítja és elmenti ugyanazt a jelenetet, az `F` billentyű folyamatos javítást kapcsol, az `R` billentyű pedig randomizálja a meglévő Gaussian-foltokat és elmenti a fájlba, az `N` billentyű pedig hozzáad egy új random Gaussian-foltot és elmenti a fájlba.
