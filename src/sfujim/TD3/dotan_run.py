import sys
import os

os.environ["CC"] = "/opt/homebrew/Cellar/gcc\@12/12.3.0/bin/gcc-12"
os.environ["CXX"] = "/opt/homebrew/Cellar/gcc\@12/12.3.0/bin/gcc-12"
os.environ["PATH"] = "/opt/homebrew/Cellar/glfw/3.3.8/lib/:" + os.environ["PATH"]
os.environ["LIBRARY_PATH"] = "/opt/homebrew/Cellar/glfw/3.3.8/lib/:"  # + os.environ["LIBRARY_PATH"]
print(os.environ["PATH"])
from src.sfujim.TD3.main import main

if __name__ == "__main__":
    sys.argv = sys.argv + ['--policy', 'TD3',
                           '--env', "HalfCheetah-v4",
                           '--seed', "1"]
    """
    python main.py \
        --policy "TD3" \
        --env "HalfCheetah-v3" \
        --seed $i
    """

    main()
