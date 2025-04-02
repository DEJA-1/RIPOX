from imageCapture.camera import CameraHandler

def select_source():
    print("Wybierz źródło obrazu:")
    print("0 - Kamera komputerowa (domyślna)")
    print("1 - Plik wideo/zdjęcie")
    print("2 - Kamera IP (RTSP/HTTP)")
    choice = input("Twój wybór (0/1/2): ").strip()

    if choice == '1':
        path = input("Podaj ścieżkę do pliku wideo/zdjęcia: ")
        return path
    elif choice == '2':
        url = input("Podaj URL kamery IP (np. rtsp://...): ")
        return url
    else:
        return 0

if __name__ == "__main__":
    source = select_source()
    camera = CameraHandler(source)
    camera.start()