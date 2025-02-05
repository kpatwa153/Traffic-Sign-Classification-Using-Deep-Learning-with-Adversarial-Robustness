<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Traffic Sign Classification Using Deep Learning with Adversarial Robustness</h3>

  <p align="center">
    In today’s world, we all are moving towards online self-driving cars and anything we can do to improve the self-driving cars will make the world even safer. We understand that a lot of companies are working on self-driving cars, but they cannot think of everything. Being a data scientist and with the opportunity given to us with this project, motivated us to work on improving the sign detections methods. 
    <br />
    <a href="https://github.com/kpatwa153/Traffic-Sign-Classification-Using-Deep-Learning-with-Adversarial-Robustness?tab=readme-ov-file"><strong>Explore the docs »</strong></a>
    <br />
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#purpose">Purpose</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li>
      <a href="#usage-instructions">Usage Instructions</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#running-the-project">Running the Project</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#known-issues">Known Issues</a></li>
    <li><a href="#feature-roadmap">Feature Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- Abstract -->
## Purpose

This project's goal is to create a **robust traffic sign categorisation system** that can recognise traffic signs correctly in both benign and hostile environments. In contemporary transportation systems, traffic sign recognition is essential, especially for advanced driver assistance systems (ADAS), driverless cars, and intelligent traffic control. However, minor, intentional changes in input data might lead to the model misclassifying signs, making classic machine learning models extremely susceptible to adversarial attacks. This presents serious safety hazards, especially for practical uses like self-driving cars.

The research employs a thorough strategy that blends conventional classification methods with adversarial defence mechanisms in order to overcome this difficulty. The Traffic Sign Dataset Classification, which includes a variety of traffic sign categories, is used to train and assess the model. The use of adversarial training methods, such as creating adversarial examples using FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent) attacks, and integrating defensive transformations like Gaussian blur, random cropping, colour jitter, and random flips are important innovations. These methods seek to improve the model's resistance to hostile perturbations and replicate real-world variability.

The project guarantees that the model not only performs well on clean data but also exhibits notable robustness against adversarially manipulated inputs by incorporating adversarial defence tactics. By lowering the likelihood of accidents and guaranteeing adherence to traffic laws, this increased dependability eventually benefits society by enhancing the safety and efficacy of AI-driven mobility solutions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [![Python]](https://docs.python.org/3/)
* [![PyTorch]](https://pytorch.org/docs/stable/index.html)
* [![Pandas]](https://pandas.pydata.org/docs/)
* [![Pillow]](https://pillow.readthedocs.io/en/stable/)
* [![Matplotlib]](https://matplotlib.org/stable/index.html)
* [![NumPy]](https://numpy.org/doc/)
* [![Torchvision]](https://pytorch.org/vision/stable/index.html)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Usage Instructions
Make sure your environment has the following versions of the APIs. 

```markdown
torch==2.0.0          # Replace with your version
torchvision==0.15.0   # Replace with your version
pandas==1.5.3         # Replace with your version
Pillow==9.4.0         # Replace with your version
matplotlib==3.7.1     # Replace with your version
numpy==1.24.2         # Replace with your version
```

### Prerequisites
Make sure you create a folder named `Data` and store the dataset there.
Dependencies can be installed by running the following command:
```sh
pip install -r requirements.txt
```

### Installation

_To run this project, follow these steps:_

1. Clone the repo
   ```sh
   git clone git@github.com:kpatwa153/Traffic-Sign-Classification-Using-Deep-Learning-with-Adversarial-Robustness.git
   ```
2. Navigate to the project directory
   ```sh
   cd fall-2024-final-project-kpatwa153
   ```
3. Install the required dependencies
   ```sh
   pip install -r requirements.txt
   ```

### Running the Project

1. Ensure you have the dataset in the `Data` folder.
2. Run the Jupyter Notebook
   ```sh
   jupyter notebook Notebook.ipynb
   ```
3. Follow the instructions in the notebook to train and evaluate the model.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Known Issues
  - Long training time on CPU; GPU recommended for better performance.
  - High memory usage during adversarial generation; reduce batch size if needed.

## Feature Roadmap
  - Implement dynamic epsilon for FGSM to analyze varying perturbation magnitudes.
  - Add PGD and CW Attack support for enhanced evaluation.
  - Develop a web-based dashboard for performance visualization.

<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

- Kenil Patwa - [@kenil-patwa-604159191](https://www.linkedin.com/in/kenil-patwa-604159191/) - kenilpatwa01@gmail.com
- Naman Singh - [@namansingh8998](https://www.linkedin.com/in/namansingh8998/) - namansingh2623@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>



