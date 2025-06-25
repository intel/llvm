// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Find version changes in data points to create annotations
 * @param {Array} points - Data points to analyze
 * @param {string} versionKey - Key to track version changes
 * @returns {Array} - List of change points
 */
function findVersionChanges(points, versionKey) {
    if (!points || points.length < 2) return [];

    const changes = [];
    // Sort points by date
    const sortedPoints = [...points].sort((a, b) => a.x - b.x);
    let lastVersion = sortedPoints[0][versionKey];

    for (let i = 1; i < sortedPoints.length; i++) {
        const currentPoint = sortedPoints[i];

        const currentVersion = currentPoint[versionKey];
        if (currentVersion && currentVersion !== lastVersion) {
            changes.push({
                date: currentPoint.x,
                newVersion: currentVersion,
            });
            lastVersion = currentVersion;
        }
    }
    
    return changes;
}

/**
 * Add version change annotations to chart options
 * @param {Object} data - Chart data
 * @param {Object} options - Chart.js options object
 */
function addVersionChangeAnnotations(data, options) {
    const benchmarkSources = Array.from(window.annotationsOptions.values());
    const changeTrackers = [
        {
            // Benchmark repos updates
            versionKey: 'gitBenchHash',
            sources: benchmarkSources,
            pointsFilter: (points, url) => points.filter(p => p.gitBenchUrl === url),
            formatLabel: (sourceName, version) => `${sourceName}: ${version.substring(0, 7)}`
        },
        {
            // Compute Runtime updates
            versionKey: 'compute_runtime',
            sources: [
                {
                    name: "Compute Runtime",
                    url: "https://github.com/intel/compute-runtime.git",
                    color: {
                        border: 'rgba(70, 105, 150, 0.8)',
                        background: 'rgba(70, 105, 150, 0.9)',
                    },
                }
            ],
        }
    ];
    
    changeTrackers.forEach(tracker => {
        tracker.sources.forEach((source) => {
            const changes = {};
            
            // Find changes across all runs
            Object.values(data.runs).flatMap(runData => 
                findVersionChanges(
                    tracker.pointsFilter ? tracker.pointsFilter(runData.data, source.url) : runData.data,
                    tracker.versionKey
                )
            ).forEach(change => {
                const changeKey = `${source.name}-${change.newVersion}`;
                if (!changes[changeKey] || change.date < changes[changeKey].date) {
                    changes[changeKey] = change;
                }
            });
            
            // Create annotation for each unique change
            Object.values(changes).forEach(change => {
                const annotationId = `${change.date}`;
                // If annotation at a given date already exists, update it
                if (options.plugins.annotation.annotations[annotationId]) {
                    options.plugins.annotation.annotations[annotationId].label.content.push(
                        tracker.formatLabel ?
                            tracker.formatLabel(source.name, change.newVersion) :
                            `${source.name}: ${change.newVersion}`
                    );
                    options.plugins.annotation.annotations[annotationId].borderColor = 'rgba(128, 128, 128, 0.8)';
                    options.plugins.annotation.annotations[annotationId].borderWidth += 1;
                    options.plugins.annotation.annotations[annotationId].label.backgroundColor = 'rgba(128, 128, 128, 0.9)';
                } else {
                    options.plugins.annotation.annotations[annotationId] = {
                        type: 'line',
                        xMin: change.date,
                        xMax: change.date,
                        borderColor: source.color.border,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        label: {
                            content: [
                                tracker.formatLabel ? 
                                tracker.formatLabel(source.name, change.newVersion) : 
                                    `${source.name}: ${change.newVersion}`
                            ],
                            display: false,
                            position: 'start',
                            backgroundColor: source.color.background,
                            z: 1,
                        }
                    }
                };
            });
        });
    });
}

/**
 * Set up event listeners for annotation interactions
 * @param {Chart} chart - Chart.js instance
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Object} options - Chart.js options object
 */
function setupAnnotationListeners(chart, ctx, options) {
    // Add event listener for annotation clicks - display/hide label
    ctx.canvas.addEventListener('click', function(e) {
        const activeElements = chart.getElementsAtEventForMode(e, 'nearest', { intersect: true }, false);
        
        // If no data point is clicked, check if an annotation was clicked
        if (activeElements.length === 0) {
            const rect = chart.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            
            // Check if click is near any annotation line
            const annotations = options.plugins.annotation.annotations;
            Object.values(annotations).some(annotation => {
                // Get the position of the annotation line
                const xPos = chart.scales.x.getPixelForValue(annotation.xMin);
                
                // Display label if click is near the annotation line (within 5 pixels)
                if (Math.abs(x - xPos) < 5) {
                    annotation.label.display = !annotation.label.display;
                    chart.update();
                    return true; // equivalent to break in a for loop
                }
                return false;
            });
        }
    });
    
    // Add mouse move handler to change cursor when hovering over annotations
    ctx.canvas.addEventListener('mousemove', function(e) {
        const rect = chart.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        
        // Check if mouse is near any annotation line
        const annotations = options.plugins.annotation.annotations;
        const isNearAnnotation = Object.values(annotations).some(annotation => {
            const xPos = chart.scales.x.getPixelForValue(annotation.xMin);
            
            if (Math.abs(x - xPos) < 5) {
                return true;
            }
            return false;
        });
        
        // Change cursor style based on proximity to annotation
        ctx.canvas.style.cursor = isNearAnnotation ? 'pointer' : '';
    });
    
    // Reset cursor when mouse leaves the chart area
    ctx.canvas.addEventListener('mouseleave', function() {
        ctx.canvas.style.cursor = '';
    });
}

// Export functions to make them available to other modules
window.ChartAnnotations = {
    findVersionChanges,
    addVersionChangeAnnotations,
    setupAnnotationListeners
};
