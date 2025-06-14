# 🤖 asimovsSpielzimmer

Welcome to **asimovsSpielzimmer** – your SO101 robot's playground! 

Yeah, we know—teleoperating your robot arm is fun and all. But let's face it—your foundation model (looking at you [smolVLA](https://huggingface.co/lerobot/smolvla_base)) still behaves like a teenager losing coordination skills during a growth spurt ([see related research – I am not a physician](https://www.biomedcentral.com/about/press-centre/science-press-releases/20-05-16)).

Yes, you’ll need a lot of data to get anywhere near generalization. Our solution takes away the chore of recording for imitation learning. Describe a task template once, generate high-variance data, duct tape it to your simulation, and record huge datasets that are out-of-the-box compatible with the [LeRobot framework](https://github.com/huggingface/lerobot).

Push it up to Huggingface 🤗 and train your robot.

![HeroImage](/materials/image.png)

---

## 🚀 What is this?

**asimovsSpielzimmer** (“Asimov’s Playroom” in German) is a toolkit designed to automatically generate diverse and realistic robot simulation configurations using OpenAI’s GPT models.  
Ideal for researchers, enthusiasts, and anyone looking to streamline the creation of large-scale, varied robot datasets fresh from the simulation press and ready to be plugged into the LeRobot Framework.

---

## ✨ Features

- **LLM-powered config generation**: Uses GPT-4o to create simulation configs from task templates.
- **Randomization**: Ensured randomness for robust and varied environments.
- **Easy extensibility**: Add your own tasks and templates.
- **Modern Pythonic codebase**: Clean, modular, and ready for hacking.
- **Aynchronous generation**: Load pregenerated configurations with your existing simulation pipelines.

---

## 🛠️ Quickstart

1. **Clone this repo**  
    ```bash
    git clone https://github.com/schutera/asimovsSpielzimmer.git
    cd asimovsSpielzimmer
    ```

2. **Install dependencies**  
    ```bash
    pip install -r requirements.txt
    ```

3. **Use existing task templates** 
    ```bash
    # Example: Generate simulation configs from the command line
    python generate_sim_config.py \
        --task_type="push_cube_to_cube" \
        --num_episodes=1
    ```

This will generate 3 simulation configs for the `push_cube_to_cube` task using the templates in `task_info.json`.  
Configs are saved automatically for each episode.

4. **Customize or add your own task templates**  
See [`task_info.json`](task_info.json) for examples.  
Add new YAML templates to extend functionality. Make sure to enable your simulation to digest the templates as well.

5. **Use in Isaac Sim**
To use the `isaac_sim_script.py` you need to download isaac sim 4.5 and run the script as standalone simulator script like this:
```
./python.sh /home/<path-to-your-script>/isaac_sim_script.py --base_yaml_dir /home/<path-to-your-config-yaml-dir>/configs --output_path /home/<your-sim-output-save-path>
```
**Quick warning:** This script was created during the 2025 Hugging Face LeRobot Hackathon and is beyond hacky and unprofessional. Use at your own risk. If you find something useful, it's probably best to pick out the good parts yourself (or with the help of your trusted LLM).


---

## 🥨 Free Data or as we call it in German: Brezel

- [Our move_to_cube imitation learning dataset](https://huggingface.co/datasets/mrkschtr/real_movetocube) 
- [Our first 50 episodes move_to_cube sim generated dataset](https://huggingface.co/datasets/RickRain/asimovsToyboxMoveToCube)
- [Move to cube Isaac Sim dataset](https://huggingface.co/datasets/RickRain/asimovsToyboxMoveToCube)
- [Pick and place Isaac Sim dataset](https://huggingface.co/datasets/RickRain/asimovsToyboxPickandplace)


---
## 📖 Citation

If you use **Asimovs Spielzimmer** in your research, please cite:

```bibtex
@misc{asimovsSpielzimmer2025,
    title        = {Asimovs Spielzimmer: Large scale simulated LeRobot datasets},
    author       = {S. Bühler and M. Schutera},
    year         = {2025},
    howpublished = {\url{https://github.com/schutera/asimovsSpielzimmer}},
    note         = {GitHub repository}
}
```
