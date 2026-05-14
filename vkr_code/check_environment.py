"""
pip install -r requirements_py310.txt
Проверка окружения перед запуском пайплайна.

Что делает:
    1. Печатает версию Python (нужно для выбора набора wheels).
    2. Печатает архитектуру и ОС.
    3. Проверяет, какие из нужных пакетов уже установлены, каких нет.
    4. Подсказывает, какую команду запускать дальше.

Использование:
    python check_environment.py
"""
import sys
import platform


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",        # импортируется как sklearn, ставится как scikit-learn
    "xgboost",
    "lifelines",
    "lifetimes",
    "shap",
    "joblib",
    "cloudpickle",
    "pyarrow",
    "scipy",
]

PACKAGE_INSTALL_NAMES = {
    "sklearn": "scikit-learn",
}


def main():
    print("=" * 60)
    print("  ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 60)

    # === Версия Python ===
    py_version = sys.version_info
    print(f"\nPython: {sys.version.split()[0]}")
    print(f"Платформа: {platform.system()} {platform.machine()}")

    # === Какой набор wheels использовать ===
    if py_version.major == 3 and py_version.minor == 10:
        wheel_dir = "wheels_py310"
        print(f"\nИспользуй: {wheel_dir}/")
    elif py_version.major == 3 and py_version.minor == 8:
        wheel_dir = "wheels_py38"
        print(f"\nИспользуй: {wheel_dir}/")
    elif py_version.major == 3 and py_version.minor in (9, 11, 12):
        # Близкие версии могут подойти, но без гарантии
        wheel_dir = None
        print(f"\n[ВНИМАНИЕ] Под Python {py_version.major}.{py_version.minor} "
              f"wheels не собраны.")
        print(f"  Попробуй wheels_py310/ — может подойти, но без гарантии.")
        print(f"  Если не сработает — нужно собрать wheels отдельно "
              f"под {py_version.major}.{py_version.minor}.")
    else:
        wheel_dir = None
        print(f"\n[ОШИБКА] Python {py_version.major}.{py_version.minor} "
              f"не поддерживается этим пакетом.")
        print(f"  Нужен Python 3.8 или 3.10.")
        return

    # === Проверка наличия пакетов ===
    print("\n" + "-" * 60)
    print("Проверка установленных пакетов:")
    print("-" * 60)

    installed = []
    missing = []

    for pkg in REQUIRED_PACKAGES:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {pkg:<15s} {version}")
            installed.append(pkg)
        except ImportError:
            install_name = PACKAGE_INSTALL_NAMES.get(pkg, pkg)
            print(f"  ✗ {pkg:<15s} НЕ УСТАНОВЛЕН (pip-имя: {install_name})")
            missing.append(install_name)

    # === Итог ===
    print("\n" + "=" * 60)
    if not missing:
        print("  ВСЕ ПАКЕТЫ УСТАНОВЛЕНЫ — можно запускать:")
        print("    python run_all.py")
    else:
        print(f"  НЕДОСТАЁТ ПАКЕТОВ: {len(missing)} из {len(REQUIRED_PACKAGES)}")
        print(f"  Не хватает: {', '.join(missing)}")
        print("\n  Установка из локальных wheels (без интернета):")
        if wheel_dir:
            req_file = f"requirements_{wheel_dir.split('_')[1]}.txt"
            print(f"    pip install --no-index --find-links={wheel_dir}/ -r {req_file}")
        else:
            print(f"    pip install --no-index --find-links=wheels_py310/ -r requirements_py310.txt")
        print("\n  Если есть доступ к корпоративному репозиторию pip:")
        print(f"    pip install -r requirements_py310.txt --index-url <URL_зеркала>")
    print("=" * 60)


if __name__ == "__main__":
    main()
