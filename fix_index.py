# Script to fix the index.html file by replacing the problematic chart function

# Read the original working template
original_chart_code = """        function updateChart(totalFields) {
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
        }"""

# The center text plugin definition
center_plugin_code = """        // Register the center text plugin globally
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

"""

print("Fix script created. The chart code has been prepared.")
print("The issue was: window.diseaseChartInstance.config.plugins.push(centerTextPlugin)")
print("Solution: Register plugin in Chart constructor with plugins: [centerTextPlugin]")
