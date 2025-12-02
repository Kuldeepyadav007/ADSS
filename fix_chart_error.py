import re

# Read the current broken file
with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where the script tag starts
script_start = content.find('<script>')
if script_start == -1:
    print("Error: Could not find script tag")
    exit(1)

# Get everything before the script tag
before_script = content[:script_start]

# Create the complete working script section
working_script = """<script>
        // Dark Mode Logic
        const themeToggle = document.getElementById('themeToggle');
        const icon = themeToggle.querySelector('i');

        if (localStorage.getItem('darkMode') === 'true') {
            document.body.classList.add('dark-mode');
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        }

        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', isDark);

            if (isDark) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            } else {
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            }
        });

        // Initialize map
        const map = L.map('fieldsMap', {
            zoomControl: false,
            attributionControl: false
        }).setView([20.5937, 78.9629], 4);

        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            maxZoom: 19
        }).addTo(map);

        // Fullscreen function
        function openFullscreenMap() {
            const mapCard = document.querySelector('.map-card');
            const btn = document.querySelector('.map-overlay-btn i');
            
            if (!document.fullscreenElement) {
                // Enter fullscreen
                if (mapCard.requestFullscreen) {
                    mapCard.requestFullscreen();
                } else if (mapCard.webkitRequestFullscreen) {
                    mapCard.webkitRequestFullscreen();
                } else if (mapCard.msRequestFullscreen) {
                    mapCard.msRequestFullscreen();
                }
                
                // Change icon to compress
                btn.classList.remove('fa-expand');
                btn.classList.add('fa-compress');
            } else {
                // Exit fullscreen
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                } else if (document.msExitFullscreen) {
                    document.msExitFullscreen();
                }
                
                // Change icon back to expand
                btn.classList.remove('fa-compress');
                btn.classList.add('fa-expand');
            }

            // Resize map after entering/exiting fullscreen
            setTimeout(() => {
                map.invalidateSize();
            }, 100);
        }
        
        // Listen for fullscreen changes
        document.addEventListener('fullscreenchange', () => {
            const btn = document.querySelector('.map-overlay-btn i');
            if (!document.fullscreenElement) {
                btn.classList.remove('fa-compress');
                btn.classList.add('fa-expand');
            }
            setTimeout(() => {
                map.invalidateSize();
            }, 100);
        });

        // Register the center text plugin globally
        const centerTextPlugin = {
            id: 'centerText',
            beforeDraw: function (chart) {
                if (chart.config.type !== 'doughnut') return;
                
                var width = chart.width,
                    height = chart.height,
                    ctx = chart.ctx;

                ctx.restore();
                var fontSize = (height / 80).toFixed(2);
                ctx.font = "bold " + fontSize + "em sans-serif";
                ctx.textBaseline = "middle";
                ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--text-main');

                var text = chart.config.options.centerText || '0',
                    textX = Math.round((width - ctx.measureText(text).width) / 2),
                    textY = height / 2;

                ctx.fillText(text, textX, textY);
                ctx.save();
            }
        };

        // Load and display fields
        async function loadFields() {
            try {
                const response = await fetch('/api/get_fields');
                const fields = await response.json();

                // Update field count and area
                const count = fields.length;
                document.getElementById('fieldCount').textContent = count;
                document.getElementById('noAdviceCount').textContent = `${count} >`;

                let totalArea = 0;
                fields.forEach(f => totalArea += (f.areaAcres || 0));
                document.getElementById('totalAreaText').textContent = `Total: ${totalArea.toFixed(1)} acre`;

                // Update chart
                updateChart(count);

                // Display fields on map with numbers
                if (fields.length > 0) {
                    const bounds = [];
                    fields.forEach((field, index) => {
                        let layer;
                        const fieldNumber = index + 1;

                        if (field.type === 'circle') {
                            layer = L.circle(field.geometry.center, {
                                radius: field.geometry.radius,
                                color: 'white',
                                weight: 2,
                                fillOpacity: 0
                            }).addTo(map);
                            bounds.push(field.geometry.center);

                            // Add field number label
                            const marker = L.marker(field.geometry.center, {
                                icon: L.divIcon({
                                    className: 'field-number-label',
                                    html: `<div style="background: rgba(0,200,83,0.95); color: white; padding: 6px 12px; border-radius: 16px; font-weight: bold; font-size: 14px; box-shadow: 0 3px 8px rgba(0,0,0,0.4); transition: all 0.2s; cursor: pointer; border: 2px solid white;">${fieldNumber}</div>`,
                                    iconSize: [40, 40]
                                })
                            }).addTo(map);
                            
                            // Click handler for marker
                            marker.on('click', () => {
                                window.location.href = `/app2?field_id=${field.id}`;
                            });

                        } else {
                            layer = L.geoJSON(field.geometry, {
                                style: {
                                    color: 'white',
                                    weight: 2,
                                    fillOpacity: 0
                                }
                            }).addTo(map);

                            const coords = field.geometry.coordinates[0];
                            coords.forEach(coord => {
                                bounds.push([coord[1], coord[0]]);
                            });

                            // Calculate center for label
                            const center = layer.getBounds().getCenter();
                            const marker = L.marker(center, {
                                icon: L.divIcon({
                                    className: 'field-number-label',
                                    html: `<div style="background: rgba(0,200,83,0.95); color: white; padding: 6px 12px; border-radius: 16px; font-weight: bold; font-size: 14px; box-shadow: 0 3px 8px rgba(0,0,0,0.4); transition: all 0.2s; cursor: pointer; border: 2px solid white;">${fieldNumber}</div>`,
                                    iconSize: [40, 40]
                                })
                            }).addTo(map);
                            
                            // Click handler for marker
                            marker.on('click', () => {
                                window.location.href = `/app2?field_id=${field.id}`;
                            });
                        }

                        // Click to navigate
                        layer.on('click', () => {
                            window.location.href = `/app2?field_id=${field.id}`;
                        });
                    });
                    map.fitBounds(bounds, { padding: [20, 20] });
                }
            } catch (error) {
                console.error('Error loading fields:', error);
            }
        }

        function updateChart(totalFields) {
            const ctx = document.getElementById('diseaseChart').getContext('2d');

            if (window.diseaseChartInstance) {
                window.diseaseChartInstance.destroy();
            }

            window.diseaseChartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['No advice', 'Other'],
                    datasets: [{
                        data: [totalFields, 0],
                        backgroundColor: ['#e0e0e0', '#ffffff'],
                        borderWidth: 0,
                        cutout: '80%'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    centerText: totalFields.toString(),
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false }
                    }
                },
                plugins: [centerTextPlugin]
            });
        }

        loadFields();
    </script>
</body>

</html>"""

# Combine everything
fixed_content = before_script + working_script

# Write the fixed file
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("✓ File fixed successfully!")
print("✓ Chart plugin now registered correctly in the Chart constructor")
print("✓ Fields should now display on the map without errors")
