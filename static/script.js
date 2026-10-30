/* ==========================================
   Face Recognition Attendance System
   script.js
========================================== */

document.addEventListener("DOMContentLoaded", function () {

    console.log("Face Recognition Attendance System Loaded");

    // =====================================
    // Auto Hide Alerts
    // =====================================

    const alerts = document.querySelectorAll(".alert");

    if (alerts.length > 0) {

        setTimeout(function () {

            alerts.forEach(function (alert) {

                alert.style.transition = "0.5s";

                alert.style.opacity = "0";

                setTimeout(function () {

                    alert.remove();

                }, 500);

            });

        }, 3000);

    }

    // =====================================
    // Delete Confirmation
    // =====================================

    const deleteButtons = document.querySelectorAll(".delete-button");

    deleteButtons.forEach(function (button) {

        button.addEventListener("click", function (e) {

            const confirmDelete = confirm(
                "Are you sure you want to delete this student?"
            );

            if (!confirmDelete) {

                e.preventDefault();

            }

        });

    });

    // =====================================
    // Button Loading Effect
    // =====================================

    const forms = document.querySelectorAll("form");

    forms.forEach(function (form) {

        form.addEventListener("submit", function () {

            const submitBtn = form.querySelector("button[type='submit']");

            if (submitBtn) {

                submitBtn.disabled = true;

                submitBtn.innerHTML =
                    '<span class="spinner-border spinner-border-sm"></span> Please Wait...';

            }

        });

    });

    // =====================================
    // Search Box Highlight
    // =====================================

    const searchInput = document.querySelector("input[name='search']");

    if (searchInput) {

        searchInput.addEventListener("focus", function () {

            this.style.boxShadow = "0 0 8px #0d6efd";

        });

        searchInput.addEventListener("blur", function () {

            this.style.boxShadow = "none";

        });

    }

    // =====================================
    // Table Row Hover
    // =====================================

    const rows = document.querySelectorAll("tbody tr");

    rows.forEach(function (row) {

        row.addEventListener("mouseenter", function () {

            this.style.transition = "0.3s";

            this.style.backgroundColor = "#f8f9fa";

        });

        row.addEventListener("mouseleave", function () {

            this.style.backgroundColor = "";

        });

    });

    // =====================================
    // Image Preview (Future Use)
    // =====================================

    const imageInput = document.querySelector("input[type='file']");

    if (imageInput) {

        imageInput.addEventListener("change", function () {

            console.log("Image Selected");

        });

    }

    // =====================================
    // Dashboard Welcome
    // =====================================

    console.log("Dashboard Ready");

});