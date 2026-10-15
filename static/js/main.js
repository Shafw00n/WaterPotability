document.addEventListener('DOMContentLoaded', function () {
  // Synchronize slider and number input
  const sliders = document.querySelectorAll('.form-slider');
  sliders.forEach(slider => {
    const targetId = slider.getAttribute('data-target');
    const numInput = document.getElementById(targetId);
    if (numInput) {
      slider.addEventListener('input', function () {
        numInput.value = this.value;
      });
      numInput.addEventListener('input', function () {
        slider.value = this.value;
      });
    }
  });

  // Preset button handling
  const presetButtons = document.querySelectorAll('.preset-card');
  presetButtons.forEach(btn => {
    btn.addEventListener('click', function () {
      const presetDataRaw = this.getAttribute('data-values');
      if (!presetDataRaw) return;

      try {
        const values = JSON.parse(presetDataRaw);
        for (const [key, val] of Object.entries(values)) {
          const numInput = document.getElementById('input_' + key);
          const slider = document.querySelector(`.form-slider[data-target="input_${key}"]`);
          if (numInput) numInput.value = val;
          if (slider) slider.value = val;
        }

        // Highlight selected card
        presetButtons.forEach(b => b.style.borderColor = 'var(--border-color)');
        this.style.borderColor = 'var(--primary)';

        // Scroll to form if on predict page
        const form = document.getElementById('prediction-form');
        if (form) {
          form.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      } catch (e) {
        console.error('Error applying preset:', e);
      }
    });
  });

  // Model switch buttons
  const modelOptions = document.querySelectorAll('.model-option');
  const modelChoiceInput = document.getElementById('model_choice_input');
  modelOptions.forEach(opt => {
    opt.addEventListener('click', function (e) {
      e.preventDefault();
      modelOptions.forEach(o => o.classList.remove('active'));
      this.classList.add('active');
      if (modelChoiceInput) {
        modelChoiceInput.value = this.getAttribute('data-model');
      }
    });
  });
});
