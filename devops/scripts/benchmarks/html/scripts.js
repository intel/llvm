// Copyright (C) 2024-2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core state
let activeRuns = new Set(defaultCompareNames);
let chartInstances = new Map();
let suiteNames = new Set();
let activeTags = new Set();
let timeseriesData, barChartsData, allRunNames;
let layerComparisonsData;
let latestRunsLookup = new Map();
let pendingCharts = new Map(); // Store chart data for lazy loading
let chartObserver; // Intersection observer for lazy loading charts
let annotationsOptions = new Map(); // Global options map for annotations
let archivedDataLoaded = false;
let loadedBenchmarkRuns = []; // Loaded results from the js/json files

// Global variables loaded from data.js:
// - benchmarkRuns: array of benchmark run data
// - benchmarkMetadata: metadata for benchmarks and groups  
// - benchmarkTags: tag definitions
// - flamegraphData: available flamegraphs (optional, added dynamically)

// DOM Elements
let runSelect, selectedRunsDiv, suiteFiltersContainer, tagFiltersContainer;

// Observer for lazy loading charts
function initChartObserver() {
    if (chartObserver) return;

    chartObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const containerId = entry.target.querySelector('canvas').id;
                if (pendingCharts.has(containerId)) {
                    const { data, type } = pendingCharts.get(containerId);
                    createChart(data, containerId, type);
                    pendingCharts.delete(containerId);
                    chartObserver.unobserve(entry.target);
                }
            }
        });
    }, {
        root: null, // viewport (current view)
        rootMargin: '100px', // Load charts a bit before they enter the viewport
        threshold: 0.1 // Start loading when 10% of the chart is within the rootMargin
    });
}

const colorPalette = [
    'rgb(255, 50, 80)',
    'rgb(255, 145, 15)',
    'rgb(255, 220, 0)',
    'rgb(20, 200, 50)',
    'rgb(0, 130, 255)',
    'rgb(180, 60, 255)',
    'rgb(255, 40, 200)',
    'rgb(0, 210, 180)',
    'rgb(255, 90, 0)',
    'rgb(110, 220, 0)',
    'rgb(240, 100, 170)',
    'rgb(30, 175, 255)',
    'rgb(180, 210, 0)',
    'rgb(130, 0, 220)',
    'rgb(255, 170, 0)',
    'rgb(0, 170, 110)',
    'rgb(220, 80, 60)',
    'rgb(80, 115, 230)',
    'rgb(210, 190, 0)',
];

const annotationPalette = [
    'rgba(167, 109, 59, 0.8)',
    'rgba(185, 185, 60, 0.8)',
    'rgba(58, 172, 58, 0.8)',
    'rgba(158, 59, 158, 0.8)',
    'rgba(167, 93, 63, 0.8)',
    'rgba(163, 60, 81, 0.8)',
    'rgba(51, 148, 155, 0.8)',
]

const nameColorMap = {};
let colorIndex = 0;

function getColorForName(name) {
    if (!(name in nameColorMap)) {
        nameColorMap[name] = colorPalette[colorIndex % colorPalette.length];
        colorIndex++;
    }
    return nameColorMap[name];
}

// Run selector functions
function updateSelectedRuns(forceUpdate = true) {
    selectedRunsDiv.innerHTML = '';
    activeRuns.forEach(name => {
        selectedRunsDiv.appendChild(createRunElement(name));
    });
    if (forceUpdate)
        updateCharts();
}

function createRunElement(name) {
    const runElement = document.createElement('span');
    runElement.className = 'selected-run';
    runElement.innerHTML = `${name} <button onclick="removeRun('${name}')">X</button>`;
    return runElement;
}

function addSelectedRun() {
    const selectedRun = runSelect.value;
    if (selectedRun && !activeRuns.has(selectedRun)) {
        activeRuns.add(selectedRun);
        updateSelectedRuns();
    }
}

function removeRun(name) {
    activeRuns.delete(name);
    updateSelectedRuns();
}

// Chart creation and update
function createChart(data, containerId, type) {
    if (chartInstances.has(containerId)) {
        chartInstances.get(containerId).destroy();
    }

    const ctx = document.getElementById(containerId).getContext('2d');

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: {
                display: true,
                text: data.display_label || data.label
            },
            subtitle: {
                display: true,
                text: data.lower_is_better ? "Lower is better" : "Higher is better"
            },
            tooltip: {
                callbacks: {
                    label: (context) => {
                        if (type === 'time') {
                            const point = context.raw;
                            return [
                                `${point.seriesName}:`,
                                `Value: ${point.y.toFixed(2)} ${data.unit}`,
                                `Stddev: ${point.stddev.toFixed(2)} ${data.unit}`,
                                `Git Hash: ${point.gitHash}`,
                                ...(point.compute_runtime && point.compute_runtime !== 'null' && point.compute_runtime !== 'unknown' ?
                                    [`Compute Runtime: ${point.compute_runtime}`] : []),
                                ...(point.gitBenchHash ? [`Bench hash: ${point.gitBenchHash.substring(0, 7)}`] : []),
                                ...(point.gitBenchUrl ? [`Bench URL: ${point.gitBenchUrl}`] : []),
                            ];
                        } else {
                            return [`${context.dataset.label}:`,
                            `Value: ${context.parsed.y.toFixed(2)} ${data.unit}`,
                            ];
                        }
                    }
                }
            },
            legend: {
                position: 'top',
                labels: {
                    boxWidth: 12,
                    padding: 10,
                }
            },
            annotation: type === 'time' ? {
                annotations: {}
            } : undefined
        },
        scales: {
            y: {
                title: {
                    display: true,
                    text: data.unit
                },
                grace: '20%',
                min: isCustomRangesEnabled() ? data.range_min : null,
                max: isCustomRangesEnabled() ? data.range_max : null
            }
        }
    };

    if (type === 'time') {
        options.interaction = {
            mode: 'nearest',
            intersect: true // Require to hover directly over a point
        };
        options.onClick = (event, elements) => {
            if (elements.length > 0) {
                const point = elements[0].element.$context.raw;
                if (point.gitHash && point.gitRepo) {
                    window.open(`https://github.com/${point.gitRepo}/commit/${point.gitHash}`, '_blank');
                }
            }
        };
        options.scales.x = {
            type: 'timeseries',
            time: {
                unit: 'day'
            },
            ticks: {
                maxRotation: 45,
                minRotation: 45,
                autoSkip: true,
                maxTicksLimit: 10
            }
        };

        // Add dependencies version change annotations
        if (Object.keys(data.runs).length > 0) {
            ChartAnnotations.addVersionChangeAnnotations(data, options);
        }
    }

    const chartConfig = {
        type: type === 'time' ? 'line' : 'bar',
        data: type === 'time' ? {
            datasets: Object.values(data.runs).map(runData => ({
                ...runData,
                // For timeseries (historical results charts) use runName,
                // otherwise use displayLabel (for layer comparison charts)
                label: containerId.startsWith('timeseries') ?
                    runData.runName :
                    (runData.displayLabel || runData.label)
            }))
        } : {
            labels: data.labels,
            datasets: data.datasets
        },
        options: options
    };

    const chart = new Chart(ctx, chartConfig);
    chartInstances.set(containerId, chart);

    // Set explicit canvas size after chart creation to ensure proper sizing
    const canvas = document.getElementById(containerId);
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // Calculate dynamic height based on number of legend items
    const legendItemCount = type === 'time' ?
        Object.values(data.runs).length :
        data.datasets.length;

    // Base chart height + legend height (25px per line + padding)
    const baseChartHeight = 350;
    const legendHeight = Math.max(legendItemCount * 25, 50); // minimum 50px for legend
    const totalHeight = baseChartHeight + legendHeight;

    // Set canvas dimensions for crisp rendering
    canvas.width = rect.width * dpr;
    canvas.height = totalHeight * dpr;

    // Scale the context to ensure correct drawing operations
    const context = canvas.getContext('2d');
    context.scale(dpr, dpr);

    // Force chart to use these exact dimensions
    chart.resize(rect.width, totalHeight);

    // Add annotation interaction handlers for time-series charts
    if (type === 'time') {
        ChartAnnotations.setupAnnotationListeners(chart, ctx, options);
    }

    return chart;
}

function updateCharts() {
    const filterRunData = (chart) => ({
        ...chart,
        runs: Object.fromEntries(
            Object.entries(chart.runs).filter(([_, data]) =>
                activeRuns.has(data.runName)
            )
        )
    });

    const filteredTimeseriesData = timeseriesData.map(filterRunData);
    const filteredLayerComparisonsData = layerComparisonsData.map(filterRunData);

    const filteredBarChartsData = barChartsData.map(chart => ({
        ...chart,
        labels: chart.labels.filter(label => activeRuns.has(label)),
        datasets: chart.datasets.map(dataset => ({
            ...dataset,
            data: dataset.data.filter((_, i) => activeRuns.has(chart.labels[i]))
        }))
    }));

    drawCharts(filteredTimeseriesData, filteredBarChartsData, filteredLayerComparisonsData);
}

function drawCharts(filteredTimeseriesData, filteredBarChartsData, filteredLayerComparisonsData) {
    // Clear existing charts
    document.querySelectorAll('.charts').forEach(container => container.innerHTML = '');
    chartInstances.forEach(chart => chart.destroy());
    chartInstances.clear();
    pendingCharts.clear();

    initChartObserver(); // For lazy loading charts

    // Create timeseries charts
    filteredTimeseriesData.forEach((data, index) => {
        const containerId = `timeseries-${index}`;
        const container = createChartContainer(data, containerId, 'benchmark');
        document.querySelector('.timeseries .charts').appendChild(container);
        
        // Only set up chart observers if not in flamegraph mode
        if (!isFlameGraphEnabled()) {
            pendingCharts.set(containerId, { data, type: 'time' });
            chartObserver.observe(container);
        }
    });

    // Create layer comparison charts
    filteredLayerComparisonsData.forEach((data, index) => {
        const containerId = `layer-comparison-${index}`;
        const container = createChartContainer(data, containerId, 'group');
        document.querySelector('.layer-comparisons .charts').appendChild(container);
        
        // Only set up chart observers if not in flamegraph mode
        if (!isFlameGraphEnabled()) {
            pendingCharts.set(containerId, { data, type: 'time' });
            chartObserver.observe(container);
        }
    });

    // Create bar charts
    filteredBarChartsData.forEach((data, index) => {
        const containerId = `barchart-${index}`;
        const container = createChartContainer(data, containerId, 'group');
        document.querySelector('.bar-charts .charts').appendChild(container);
        
        // Only set up chart observers if not in flamegraph mode
        if (!isFlameGraphEnabled()) {
            pendingCharts.set(containerId, { data, type: 'bar' });
            chartObserver.observe(container);
        }
    });

    // Apply current filters
    filterCharts();
}

function createChartContainer(data, canvasId, type) {
    const container = document.createElement('div');
    container.className = 'chart-container';
    container.setAttribute('data-label', data.label);
    container.setAttribute('data-suite', data.suite);

    // Create header section for metadata
    const headerSection = document.createElement('div');
    headerSection.className = 'chart-header';

    // Check if this benchmark is marked as unstable
    const metadata = metadataForLabel(data.label, type);
    if (metadata && metadata.unstable) {
        container.setAttribute('data-unstable', 'true');

        // Add unstable warning
        const unstableWarning = document.createElement('div');
        unstableWarning.className = 'benchmark-unstable';
        unstableWarning.textContent = metadata.unstable;
        unstableWarning.style.display = isUnstableEnabled() ? 'block' : 'none';
        unstableWarning.style.marginBottom = '5px';
        headerSection.appendChild(unstableWarning);
    }

    // Add description if present in metadata
    if (metadata && metadata.description) {
        const descElement = document.createElement('div');
        descElement.className = 'benchmark-description';
        descElement.textContent = metadata.description;
        descElement.style.marginBottom = '5px';
        headerSection.appendChild(descElement);
    }

    // Add notes if present
    if (metadata && metadata.notes) {
        const noteElement = document.createElement('div');
        noteElement.className = 'benchmark-note';
        noteElement.textContent = metadata.notes;
        noteElement.style.display = isNotesEnabled() ? 'block' : 'none';
        headerSection.appendChild(noteElement);
    }

    // Add tags if present
    if (metadata && metadata.tags) {
        container.setAttribute('data-tags', metadata.tags.join(','));

        // Add tags display
        const tagsContainer = document.createElement('div');
        tagsContainer.className = 'benchmark-tags';

        metadata.tags.forEach(tag => {
            const tagElement = document.createElement('span');
            tagElement.className = 'tag';
            tagElement.textContent = tag;
            tagElement.setAttribute('data-tag', tag);

            // Add tooltip with tag description
            if (benchmarkTags[tag]) {
                tagElement.setAttribute('title', benchmarkTags[tag].description);
            }

            tagsContainer.appendChild(tagElement);
        });

        headerSection.appendChild(tagsContainer);
    }

    // Add header section to container
    container.appendChild(headerSection);

    // Create main content section (chart + legend area)
    const contentSection = document.createElement('div');
    contentSection.className = 'chart-content';

    // Check if flamegraph mode is enabled
    if (isFlameGraphEnabled()) {
        // Get all flamegraph data for this benchmark from selected runs
        const flamegraphsToShow = getFlameGraphsForBenchmark(data.label, activeRuns);
        
        if (flamegraphsToShow.length > 0) {
            // Create multiple iframes for each run that has flamegraph data
            flamegraphsToShow.forEach((flamegraphInfo, index) => {
                const iframe = document.createElement('iframe');
                iframe.src = flamegraphInfo.path;
                
                // Calculate dimensions that fit within the existing container constraints
                // The container has max-width: 1100px with 24px padding on each side
                const containerMaxWidth = 1100;
                const containerPadding = 48; // 24px on each side
                const availableWidth = containerMaxWidth - containerPadding;
                
                // Set dimensions to fit within container without scrollbars
                iframe.style.width = '100%';
                iframe.style.maxWidth = `${availableWidth}px`;
                iframe.style.height = '600px';
                iframe.style.border = '1px solid #ddd';
                iframe.style.borderRadius = '4px';
                iframe.style.display = 'block';
                iframe.style.margin = index === 0 ? '0 auto 10px auto' : '10px auto'; // Add spacing between multiple iframes
                iframe.title = `${flamegraphInfo.runName} - ${data.label}`;
                
                // Add error handling for missing flamegraph files
                iframe.onerror = function() {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'flamegraph-error';
                    errorDiv.textContent = `No flamegraph available for ${flamegraphInfo.runName} - ${data.label}`;
                    contentSection.replaceChild(errorDiv, iframe);
                };
                
                contentSection.appendChild(iframe);
            });
            
            // Add resize handling to maintain proper sizing for all iframes
            const updateIframeSizes = () => {
                const containerMaxWidth = 1100;
                const containerPadding = 48;
                const availableWidth = containerMaxWidth - containerPadding;
                
                contentSection.querySelectorAll('iframe[src*="flamegraphs"]').forEach(iframe => {
                    iframe.style.maxWidth = `${availableWidth}px`;
                });
            };
            
            // Update size on window resize
            window.addEventListener('resize', updateIframeSizes);
        } else {
            // Show message when no flamegraph is available
            const noFlameGraphDiv = document.createElement('div');
            noFlameGraphDiv.className = 'flamegraph-unavailable';
            noFlameGraphDiv.textContent = `No flamegraph data available for ${data.label}`;
            contentSection.appendChild(noFlameGraphDiv);
        }
    } else {
        // Canvas for the chart - fixed position in content flow
        const canvas = document.createElement('canvas');
        canvas.id = canvasId;
        canvas.style.width = '100%';

        // Set a default height - will be properly sized later in createChart
        canvas.style.height = '400px';
        canvas.style.marginBottom = '10px';
        contentSection.appendChild(canvas);
    }

    container.appendChild(contentSection);

    // Create footer section for details
    const footerSection = document.createElement('div');
    footerSection.className = 'chart-footer';

    // Create details section for extra info
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = "Details";

    // Add subtle download button to the summary
    const downloadButton = document.createElement('button');
    downloadButton.className = 'download-button';
    downloadButton.textContent = 'Download';
    downloadButton.onclick = (event) => {
        event.stopPropagation(); // Prevent details toggle
        downloadChart(canvasId, data.label);
    };
    summary.appendChild(downloadButton);
    details.appendChild(summary);

    // Create and append extra info
    const extraInfo = document.createElement('div');
    extraInfo.className = 'extra-info';
    extraInfo.innerHTML = generateExtraInfo(data, 'benchmark');
    details.appendChild(extraInfo);

    footerSection.appendChild(details);
    container.appendChild(footerSection);

    return container;
}

function metadataForLabel(label, type) {
    // First try exact match
    if (benchmarkMetadata[label]?.type === type) {
        return benchmarkMetadata[label];
    }

    // Then fall back to prefix match for backward compatibility
    for (const [key, metadata] of Object.entries(benchmarkMetadata)) {
        if (metadata.type === type && label.startsWith(key)) {
            return metadata;
        }
    }
    return null;
}

// Pre-compute a lookup for the latest run per label
function createLatestRunsLookup() {
    const latestRunsMap = new Map();

    loadedBenchmarkRuns.forEach(run => {
        const runDate = run.date;
        run.results.forEach(result => {
            const label = result.label;
            if (!latestRunsMap.has(label) || runDate > latestRunsMap.get(label).date) {
                latestRunsMap.set(label, {
                    run,
                    result
                });
            }
        });
    });

    return latestRunsMap;
}

function extractLabels(data) {
    // For layer comparison charts
    if (data.benchmarkLabels) {
        return data.benchmarkLabels;
    }

    // For bar charts
    if (data.datasets) {
        // Use the unique lookupLabel for filtering and lookup purposes
        return data.datasets.map(dataset => dataset.lookupLabel || dataset.label);
    }

    // For time series charts
    return [data.label];
}

function getDisplayLabel(label, data, metadata) {
    if (data.datasets) {
        // For bar charts, find the corresponding dataset and use its display label
        const dataset = data.datasets.find(d => (d.lookupLabel || d.label) === label);
        if (dataset) {
            return dataset.label;
        }
    } else if (metadata && metadata.display_name) {
        // For other chart types
        return metadata.display_name;
    }
    return label;
}

function generateExtraInfo(data, type = 'benchmark') {
    const labels = extractLabels(data);

    return labels.map(label => {
        const metadata = metadataForLabel(label, type);
        const latestRun = latestRunsLookup.get(label);
        const displayLabel = getDisplayLabel(label, data, metadata);

        let html = '<div class="extra-info-entry">';

        if (metadata && latestRun) {
            html += `<strong>${displayLabel}:</strong> ${formatCommand(latestRun.result)}<br>`;

            if (metadata.description) {
                html += `<em>Description:</em> ${metadata.description}`;
            }

            if (metadata.notes) {
                html += `<br><em>Notes:</em> <span class="note-text">${metadata.notes}</span>`;
            }

            if (metadata.unstable) {
                html += `<br><em class="unstable-warning">⚠️ Unstable:</em> <span class="unstable-text">${metadata.unstable}</span>`;
            }
        } else {
            html += `<strong>${displayLabel}:</strong> No data available`;
        }

        html += '</div>';
        return html;
    }).join('');
}

function formatCommand(run) {
    const envVars = Object.entries(run.env || {}).map(([key, value]) => `${key}=${value}`).join(' ');
    let command = run.command ? [...run.command] : [];

    return `${envVars} ${command.join(' ')}`.trim();
}

function downloadChart(canvasId, label) {
    const chart = chartInstances.get(canvasId);
    if (chart) {
        const link = document.createElement('a');
        link.href = chart.toBase64Image('image/png', 1)
        link.download = `${label}.png`;
        link.click();
    }
}

// URL and filtering functions
//
// Information about currently displayed charts, filters, etc. are preserved in
// the URL query string: This allows users to save/share links reproducing exact
// queries, filters, settings, etc. Therefore, for consistency, the URL needs to
// be reconstruted everytime queries, filters, etc. are changed.

function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

function updateURL() {
    const url = new URL(window.location);
    const regex = document.getElementById('bench-filter').value;
    const activeSuites = getActiveSuites();
    const activeRunsList = Array.from(activeRuns);
    const activeTagsList = Array.from(activeTags);

    if (regex) {
        url.searchParams.set('regex', regex);
    } else {
        url.searchParams.delete('regex');
    }

    if (activeSuites.length > 0 && activeSuites.length != suiteNames.size) {
        url.searchParams.set('suites', activeSuites.join(','));
    } else {
        url.searchParams.delete('suites');
    }

    // Add tags to URL
    if (activeTagsList.length > 0) {
        url.searchParams.set('tags', activeTagsList.join(','));
    } else {
        url.searchParams.delete('tags');
    }

    // Handle the runs parameter
    if (activeRunsList.length > 0) {
        // Check if the active runs are the same as default runs
        const defaultRuns = new Set(defaultCompareNames || []);
        const isDefaultRuns = activeRunsList.length === defaultRuns.size &&
            activeRunsList.every(run => defaultRuns.has(run));

        if (isDefaultRuns) {
            // If it's just the default runs, omit the parameter entirely
            url.searchParams.delete('runs');
        } else {
            url.searchParams.set('runs', activeRunsList.join(','));
        }
    } else {
        url.searchParams.delete('runs');
    }

    // Add toggle states to URL
    if (isNotesEnabled()) {
        url.searchParams.delete('notes');
    } else {
        url.searchParams.set('notes', 'false');
    }

    if (!isUnstableEnabled()) {
        url.searchParams.delete('unstable');
    } else {
        url.searchParams.set('unstable', 'true');
    }

    if (!isCustomRangesEnabled()) {
        url.searchParams.delete('customRange');
    } else {
        url.searchParams.set('customRange', 'true');
    }

    if (!isArchivedDataEnabled()) {
        url.searchParams.delete('archived');
    } else {
        url.searchParams.set('archived', 'true');
    }

    if (!isFlameGraphEnabled()) {
        url.searchParams.delete('flamegraph');
    } else {
        url.searchParams.set('flamegraph', 'true');
    }

    history.replaceState(null, '', url);
}

function filterCharts() {
    const regexInput = document.getElementById('bench-filter').value;
    const regex = new RegExp(regexInput, 'i');
    const activeSuites = getActiveSuites();

    document.querySelectorAll('.chart-container').forEach(container => {
        const label = container.getAttribute('data-label');
        const suite = container.getAttribute('data-suite');
        const isUnstable = container.getAttribute('data-unstable') === 'true';
        const tags = container.getAttribute('data-tags') ?
            container.getAttribute('data-tags').split(',') : [];

        // Check if benchmark has all active tags (if any are selected)
        const hasAllActiveTags = activeTags.size === 0 ||
            Array.from(activeTags).every(tag => tags.includes(tag));

        // Hide unstable benchmarks if showUnstable is false
        const shouldShow = regex.test(label) &&
            activeSuites.includes(suite) &&
            (isUnstableEnabled() || !isUnstable) &&
            hasAllActiveTags;

        container.style.display = shouldShow ? '' : 'none';
    });

    updateURL();
}

function getActiveSuites() {
    return Array.from(document.querySelectorAll('.suite-checkbox:checked'))
        .map(checkbox => checkbox.getAttribute('data-suite'));
}

// Data processing
function processTimeseriesData() {
    const resultsByLabel = {};

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            const metadata = metadataForLabel(result.label, 'benchmark');

            if (!resultsByLabel[result.label]) {
                resultsByLabel[result.label] = {
                    label: result.label,
                    display_label: metadata?.display_name || result.label,
                    suite: result.suite,
                    unit: result.unit,
                    lower_is_better: result.lower_is_better,
                    range_min: metadata?.range_min ?? null, // can't use || because js treats 0 as null
                    range_max: metadata?.range_max ?? null,
                    runs: {}
                };
            }
            addRunDataPoint(resultsByLabel[result.label], run, result, false, run.name);
        });
    });

    return Object.values(resultsByLabel);
}

function processBarChartsData() {
    const groupedResults = {};

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            const resultMetadata = metadataForLabel(result.label, 'benchmark');
            const explicitGroup = resultMetadata?.explicit_group || result?.explicit_group;
            if (!explicitGroup) return;

            if (!groupedResults[explicitGroup]) {
                // Look up group metadata
                const groupMetadata = metadataForLabel(explicitGroup, 'group');

                groupedResults[explicitGroup] = {
                    label: explicitGroup,
                    display_label: groupMetadata?.display_name || explicitGroup, // Use display_name if available
                    suite: result.suite,
                    unit: result.unit,
                    lower_is_better: result.lower_is_better,
                    range_min: groupMetadata?.range_min ?? null, // can't use || because js treats 0 as null
                    range_max: groupMetadata?.range_max ?? null,
                    labels: [],
                    datasets: [],
                    // Add metadata if available
                    description: groupMetadata?.description || null,
                    notes: groupMetadata?.notes || null,
                    unstable: groupMetadata?.unstable || null
                };
            }

            const group = groupedResults[explicitGroup];

            if (!group.labels.includes(run.name)) {
                group.labels.push(run.name);
            }

            // Store the label we'll use for lookup and the display label separately
            const lookupLabel = result.label;
            // First try to get display name from metadata using the actual label
            const metadata = benchmarkMetadata[result.label];
            const displayLabel = metadata?.display_name || result.label;

            let dataset = group.datasets.find(d => d.lookupLabel === lookupLabel);
            if (!dataset) {
                const datasetIndex = group.datasets.length;
                dataset = {
                    lookupLabel: lookupLabel, // Store the original label for lookup
                    label: displayLabel,      // Use display label for rendering
                    data: new Array(group.labels.length).fill(null),
                    backgroundColor: colorPalette[datasetIndex % colorPalette.length],
                    borderColor: colorPalette[datasetIndex % colorPalette.length],
                    borderWidth: 1
                };
                group.datasets.push(dataset);
            }

            const runIndex = group.labels.indexOf(run.name);
            if (dataset.data[runIndex] == null)
                dataset.data[runIndex] = result.value;
        });
    });

    return Object.values(groupedResults);
}

function getLayerTags(metadata) {
    const layerTags = new Set();
    if (metadata?.tags) {
        metadata.tags.forEach(tag => {
            if (tag.startsWith('SYCL') || tag.startsWith('UR') || tag === 'L0') {
                layerTags.add(tag);
            }
        });
    }
    return layerTags;
}

function processLayerComparisonsData() {
    const groupedResults = {};
    const labelsByGroup = {};

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            const resultMetadata = metadataForLabel(result.label, 'benchmark');
            const explicitGroup = resultMetadata?.explicit_group || result.explicit_group;
            if (!explicitGroup) return;

            if (!labelsByGroup[explicitGroup]) {
                labelsByGroup[explicitGroup] = new Set();
            }
            labelsByGroup[explicitGroup].add(result.label);
        });
    });

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            // Get explicit_group from metadata
            const resultMetadata = metadataForLabel(result.label, 'benchmark');
            const explicitGroup = resultMetadata?.explicit_group || result.explicit_group;
            if (!explicitGroup) return;

            // Skip if no metadata available
            const metadata = metadataForLabel(explicitGroup, 'group');
            if (!metadata) return;

            // Get all benchmark labels in this group
            const labelsInGroup = labelsByGroup[explicitGroup];

            // Check if this group compares different layers
            const uniqueLayers = new Set();
            labelsInGroup.forEach(label => {
                const labelMetadata = metadataForLabel(label, 'benchmark');
                const layerTags = getLayerTags(labelMetadata);
                layerTags.forEach(tag => uniqueLayers.add(tag));
            });

            // Only process groups that compare different layers
            if (uniqueLayers.size <= 1) return;

            if (!groupedResults[explicitGroup]) {
                groupedResults[explicitGroup] = {
                    label: explicitGroup,
                    suite: result.suite,
                    unit: result.unit,
                    lower_is_better: result.lower_is_better,
                    range_min: metadata?.range_min ?? null, // can't use || because js treats 0 as null
                    range_max: metadata?.range_max ?? null,
                    runs: {},
                    benchmarkLabels: [],
                    description: metadata?.description || null,
                    notes: metadata?.notes || null,
                    unstable: metadata?.unstable || null
                };
            }

            const group = groupedResults[explicitGroup];
            const name = result.label + ' (' + run.name + ')';

            // Add the benchmark label if it's not already in the array
            if (!group.benchmarkLabels.includes(result.label)) {
                group.benchmarkLabels.push(result.label);
            }

            addRunDataPoint(group, run, result, true, name);
        });
    });

    return Object.values(groupedResults);
}

function addRunDataPoint(group, run, result, comparison, name = null) {
    const runKey = name || result.label + ' (' + run.name + ')';

    if (!group.runs[runKey]) {
        const datasetIndex = Object.keys(group.runs).length;
        const metadata = benchmarkMetadata[result.name];
        const displayName = metadata?.display_name || result.label;
        group.runs[runKey] = {
            label: runKey,
            displayLabel: displayName + ' (' + run.name + ')', // Format for layer comparison charts
            runName: run.name,
            data: [],
            borderColor:
                comparison ? colorPalette[datasetIndex % colorPalette.length] : getColorForName(run.name),
            backgroundColor:
                comparison ? colorPalette[datasetIndex % colorPalette.length] : getColorForName(run.name),
            borderWidth: 1,
            pointRadius: 3,
            pointStyle: 'circle',
            pointHoverRadius: 5
        };
    }

    group.runs[runKey].data.push({
        // For historical results use only run.name, for layer comparisons use displayLabel
        seriesName: name === run.name ? run.name : group.runs[runKey].displayLabel,
        x: new Date(run.date),
        y: result.value,
        stddev: result.stddev,
        gitHash: run.git_hash,
        gitRepo: run.github_repo,
        compute_runtime: run.compute_runtime,
        gitBenchUrl: result.git_url,
        gitBenchHash: result.git_hash,
    });

    return group;
}

// Setup functions
function setupRunSelector() {
    runSelect = document.getElementById('run-select');
    selectedRunsDiv = document.getElementById('selected-runs');

    // Clear existing options first to prevent duplicates when reloading with archived data
    runSelect.innerHTML = '';

    allRunNames.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        runSelect.appendChild(option);
    });

    updateSelectedRuns(false);
}

function setupSuiteFilters() {
    suiteFiltersContainer = document.getElementById('suite-filters');

    // Clear existing suite filters before adding new ones
    suiteFiltersContainer.innerHTML = '';

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            suiteNames.add(result.suite);
        });
    });

    // Debug logging for suite names
    console.log('Available suites:', Array.from(suiteNames));
    console.log('Loaded benchmark runs:', loadedBenchmarkRuns.map(run => ({
        name: run.name,
        suites: [...new Set(run.results.map(r => r.suite))]
    })));

    suiteNames.forEach(suite => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'suite-checkbox';
        checkbox.dataset.suite = suite;
        checkbox.checked = true;
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(' ' + suite));
        suiteFiltersContainer.appendChild(label);
        suiteFiltersContainer.appendChild(document.createTextNode(' '));
    });
}

function isNotesEnabled() {
    const notesToggle = document.getElementById('show-notes');
    return notesToggle.checked;
}

function isUnstableEnabled() {
    const unstableToggle = document.getElementById('show-unstable');
    return unstableToggle.checked;
}

function isCustomRangesEnabled() {
    const rangesToggle = document.getElementById('custom-range');
    return rangesToggle.checked;
}

function isArchivedDataEnabled() {
    const archivedDataToggle = document.getElementById('show-archived-data');
    return archivedDataToggle.checked;
}

function isFlameGraphEnabled() {
    const flameGraphToggle = document.getElementById('show-flamegraph');
    return flameGraphToggle.checked;
}

function validateFlameGraphData() {
    return window.flamegraphData?.runs !== undefined;
}

function createFlameGraphPath(benchmarkLabel, runName, timestamp) {
    const benchmarkDirName = benchmarkLabel;
    const timestampPrefix = timestamp + '_';
    return `results/flamegraphs/${encodeURIComponent(benchmarkDirName)}/${timestampPrefix}${runName}.svg`;
}

function getRunsWithFlameGraph(benchmarkLabel, activeRuns) {
    // Inline validation for better performance
    if (!window.flamegraphData?.runs) {
        return [];
    }
    
    const runsWithFlameGraph = [];
    activeRuns.forEach(runName => {
        if (flamegraphData.runs[runName] && 
            flamegraphData.runs[runName].benchmarks && 
            flamegraphData.runs[runName].benchmarks.includes(benchmarkLabel)) {
            runsWithFlameGraph.push({
                name: runName,
                timestamp: flamegraphData.runs[runName].timestamp
            });
        }
    });
    
    return runsWithFlameGraph;
}

// Removed: getFlameGraphPath() - functionality consolidated into getFlameGraphsForBenchmark()

function getFlameGraphsForBenchmark(benchmarkLabel, activeRuns) {
    const runsWithFlameGraph = getRunsWithFlameGraph(benchmarkLabel, activeRuns);
    const flamegraphsToShow = [];
    
    // For each run that has flamegraph data, create the path
    runsWithFlameGraph.forEach(runInfo => {
        const flamegraphPath = createFlameGraphPath(benchmarkLabel, runInfo.name, runInfo.timestamp);
        
        flamegraphsToShow.push({
            path: flamegraphPath,
            runName: runInfo.name,
            timestamp: runInfo.timestamp
        });
    });
    
    // Sort by the order of activeRuns to maintain consistent display order
    const runOrder = Array.from(activeRuns);
    flamegraphsToShow.sort((a, b) => {
        const indexA = runOrder.indexOf(a.runName);
        const indexB = runOrder.indexOf(b.runName);
        return indexA - indexB;
    });
    
    return flamegraphsToShow;
}

// Removed: getFlameGraphInfo() - unused function, functionality covered by getFlameGraphsForBenchmark()

function updateFlameGraphTooltip() {
    const flameGraphToggle = document.getElementById('show-flamegraph');
    const label = document.querySelector('label[for="show-flamegraph"]');
    
    if (!flameGraphToggle || !label) return;
    
    // Check if we have flamegraph data
    if (validateFlameGraphData()) {
        const runsWithFlameGraphs = Object.keys(flamegraphData.runs).filter(
            runName => flamegraphData.runs[runName].benchmarks && 
                      flamegraphData.runs[runName].benchmarks.length > 0
        );
        
        if (runsWithFlameGraphs.length > 0) {
            label.title = `Show flamegraph SVG files instead of benchmark charts. Available for runs: ${runsWithFlameGraphs.join(', ')}`;
            flameGraphToggle.disabled = false;
            label.style.color = '';
        } else {
            label.title = 'No flamegraph data available - run benchmarks with --flamegraph option to enable';
            flameGraphToggle.disabled = true;
            label.style.color = '#999';
        }
    } else {
        label.title = 'No flamegraph data available - run benchmarks with --flamegraph option to enable';
        flameGraphToggle.disabled = true;
        label.style.color = '#999';
    }
}

function setupToggles() {
    const notesToggle = document.getElementById('show-notes');
    const unstableToggle = document.getElementById('show-unstable');
    const customRangeToggle = document.getElementById('custom-range');
    const archivedDataToggle = document.getElementById('show-archived-data');
    const flameGraphToggle = document.getElementById('show-flamegraph');

    notesToggle.addEventListener('change', function () {
        // Update all note elements visibility
        document.querySelectorAll('.benchmark-note').forEach(note => {
            note.style.display = isNotesEnabled() ? 'block' : 'none';
        });
        updateURL();
    });

    unstableToggle.addEventListener('change', function () {
        // Update all unstable warning elements visibility
        document.querySelectorAll('.benchmark-unstable').forEach(warning => {
            warning.style.display = isUnstableEnabled() ? 'block' : 'none';
        });
        filterCharts();
    });

    customRangeToggle.addEventListener('change', function () {
        // redraw all charts
        updateCharts();
    });

    // Add event listener for flamegraph toggle
    if (flameGraphToggle) {
        flameGraphToggle.addEventListener('change', function() {
            updateCharts();
            updateURL();
        });
        
        // Update flamegraph toggle tooltip with run information
        updateFlameGraphTooltip();
    }

    // Add event listener for archived data toggle
    if (archivedDataToggle) {
        archivedDataToggle.addEventListener('change', function() {
            if (archivedDataToggle.checked) {
                loadArchivedData();
            } else {
                if (archivedDataLoaded) {
                    // Reload the page to reset
                    location.reload();
                }
            }
            updateURL();
        });
    }

    // Initialize from URL params if present
    const notesParam = getQueryParam('notes');
    const unstableParam = getQueryParam('unstable');
    const archivedParam = getQueryParam('archived');
    const flamegraphParam = getQueryParam('flamegraph');

    if (notesParam !== null) {
        let showNotes = notesParam === 'true';
        notesToggle.checked = showNotes;
    }

    if (unstableParam !== null) {
        let showUnstable = unstableParam === 'true';
        unstableToggle.checked = showUnstable;
    }

    const customRangesParam = getQueryParam('customRange');
    if (customRangesParam !== null) {
        customRangeToggle.checked = customRangesParam === 'true';
    }

    if (flameGraphToggle && flamegraphParam !== null) {
        flameGraphToggle.checked = flamegraphParam === 'true';
    }

    if (archivedDataToggle && archivedParam !== null) {
        archivedDataToggle.checked = archivedParam === 'true';

        if (archivedDataToggle.checked) {
            loadArchivedData();
        }
    }
}

function setupTagFilters() {
    tagFiltersContainer = document.getElementById('tag-filters');

    // Clear existing tag filters before adding new ones
    tagFiltersContainer.innerHTML = '';

    const allTags = [];

    if (benchmarkTags) {
        for (const tag in benchmarkTags) {
            if (!allTags.includes(tag)) {
                allTags.push(tag);
            }
        }
    }

    // Create tag filter elements
    allTags.forEach(tag => {
        const tagContainer = document.createElement('div');
        tagContainer.className = 'tag-filter';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `tag-${tag}`;
        checkbox.className = 'tag-checkbox';
        checkbox.dataset.tag = tag;

        const label = document.createElement('label');
        label.htmlFor = `tag-${tag}`;
        label.textContent = tag;

        // Add info icon with tooltip if tag description exists
        if (benchmarkTags[tag]) {
            const infoIcon = document.createElement('span');
            infoIcon.className = 'tag-info';
            infoIcon.textContent = 'ⓘ';
            infoIcon.title = benchmarkTags[tag].description;
            label.appendChild(infoIcon);
        }

        checkbox.addEventListener('change', function () {
            if (this.checked) {
                activeTags.add(tag);
            } else {
                activeTags.delete(tag);
            }
            filterCharts();
        });

        tagContainer.appendChild(checkbox);
        tagContainer.appendChild(label);
        tagFiltersContainer.appendChild(tagContainer);
    });
}

function toggleAllTags(select) {
    const checkboxes = document.querySelectorAll('.tag-checkbox');

    checkboxes.forEach(checkbox => {
        checkbox.checked = select;
        const tag = checkbox.dataset.tag;

        if (select) {
            activeTags.add(tag);
        } else {
            activeTags.delete(tag);
        }
    });

    filterCharts();
}

function initializeCharts() {
    console.log('initializeCharts() started');
    
    // Process raw data
    console.log('Processing timeseries data...');
    timeseriesData = processTimeseriesData();
    console.log('Timeseries data processed:', timeseriesData.length, 'items');
    
    console.log('Processing bar charts data...');
    barChartsData = processBarChartsData();
    console.log('Bar charts data processed:', barChartsData.length, 'items');
    
    console.log('Processing layer comparisons data...');
    layerComparisonsData = processLayerComparisonsData();
    console.log('Layer comparisons data processed:', layerComparisonsData.length, 'items');
    
    allRunNames = [...new Set(loadedBenchmarkRuns.map(run => run.name))];
    latestRunsLookup = createLatestRunsLookup();
    console.log('Run names and lookup created. Runs:', allRunNames);

    // Check if we have actual benchmark results vs flamegraph-only results
    const hasActualBenchmarks = loadedBenchmarkRuns.some(run => 
        run.results && run.results.some(result => result.suite !== 'flamegraph')
    );
    
    const hasFlameGraphResults = loadedBenchmarkRuns.some(run => 
        run.results && run.results.some(result => result.suite === 'flamegraph')
    ) || (validateFlameGraphData() && Object.keys(flamegraphData.runs).length > 0);

    console.log('Benchmark analysis:', {
        hasActualBenchmarks,
        hasFlameGraphResults,
        loadedBenchmarkRuns: loadedBenchmarkRuns.length
    });

    // If we only have flamegraph results (no actual benchmark data), create synthetic data
    if (!hasActualBenchmarks && hasFlameGraphResults) {
        console.log('Detected flamegraph-only mode - creating synthetic data for flamegraphs');
        
        // Check if we have flamegraph data available
        const hasFlamegraphData = validateFlameGraphData() && 
                                 Object.keys(flamegraphData.runs).length > 0 &&
                                 Object.values(flamegraphData.runs).some(run => run.benchmarks && run.benchmarks.length > 0);
        
        if (hasFlamegraphData) {
            console.log('Creating synthetic benchmark data for flamegraph display');
            createFlameGraphOnlyData();
            
            // Auto-enable flamegraph mode for user convenience
            const flameGraphToggle = document.getElementById('show-flamegraph');
            if (flameGraphToggle && !flameGraphToggle.checked) {
                flameGraphToggle.checked = true;
                console.log('Auto-enabled flamegraph view for flamegraph-only data');
            }
        } else {
            console.log('No flamegraph data available - showing message');
            displayNoDataMessage();
        }
    } else if (!hasActualBenchmarks && !hasFlameGraphResults) {
        // No runs and no results - something went wrong
        console.log('No benchmark data found at all');
        displayNoDataMessage();
    }

    // Create global options map for annotations
    annotationsOptions = createAnnotationsOptions();
    // Make it available to the ChartAnnotations module
    window.annotationsOptions = annotationsOptions;

    // Set up active runs
    const runsParam = getQueryParam('runs');
    if (runsParam) {
        const runsFromUrl = runsParam.split(',');

        // Start with an empty set
        activeRuns = new Set();

        // Process each run from URL
        runsFromUrl.forEach(run => {
            if (run === 'default') {
                // Special case: include all default runs
                (defaultCompareNames || []).forEach(defaultRun => {
                    if (allRunNames.includes(defaultRun)) {
                        activeRuns.add(defaultRun);
                    }
                });
            } else if (allRunNames.includes(run)) {
                // Add the specific run if it exists
                activeRuns.add(run);
            }
        });
    } else {
        // No runs parameter, use defaults
        activeRuns = new Set(defaultCompareNames || []);
    }

    // Setup UI components
    setupRunSelector();
    setupSuiteFilters();
    setupTagFilters();
    setupToggles();

    // Apply URL parameters
    const regexParam = getQueryParam('regex');
    const suitesParam = getQueryParam('suites');
    const tagsParam = getQueryParam('tags');

    if (regexParam) {
        document.getElementById('bench-filter').value = regexParam;
    }

    if (suitesParam) {
        const suites = suitesParam.split(',');
        document.querySelectorAll('.suite-checkbox').forEach(checkbox => {
            checkbox.checked = suites.includes(checkbox.getAttribute('data-suite'));
        });
    }

    // Apply tag filters from URL
    if (tagsParam) {
        const tags = tagsParam.split(',');
        tags.forEach(tag => {
            const checkbox = document.querySelector(`.tag-checkbox[data-tag="${tag}"]`);
            if (checkbox) {
                checkbox.checked = true;
                activeTags.add(tag);
            }
        });
    }

    // Setup event listeners
    document.querySelectorAll('.suite-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', filterCharts);
    });
    document.getElementById('bench-filter').addEventListener('input', filterCharts);

    // Draw initial charts
    updateCharts();
}

// Make functions available globally for onclick handlers
window.addSelectedRun = addSelectedRun;
window.removeRun = removeRun;
window.toggleAllTags = toggleAllTags;

// Helper function to fetch and process benchmark data
function fetchAndProcessData(url, isArchived = false) {
    const loadingIndicator = document.getElementById('loading-indicator');

    return fetch(url)
        .then(response => {
            if (!response.ok) { throw new Error(`Got response status ${response.status}.`) }
            return response.json();
        })
        .then(data => {
            const newRuns = data.runs || data;

            if (isArchived) {
                // Merge with existing data for archived data
                loadedBenchmarkRuns = loadedBenchmarkRuns.concat(newRuns);
                archivedDataLoaded = true;
            } else {
                // Replace existing data for current data
                loadedBenchmarkRuns = newRuns;
            }
            // The following variables have same values regardless of whether
            // we load archived or current data
            benchmarkMetadata = data.metadata || benchmarkMetadata || {};
            benchmarkTags = data.tags || benchmarkTags || {};

            initializeCharts();
        })
        .catch(error => {
            console.error(`Error fetching ${isArchived ? 'archived' : 'remote'} data:`, error);
            loadingIndicator.textContent = 'Fetching remote data failed.';
        })
        .finally(() => {
            loadingIndicator.style.display = 'none';
        });
}

// Load data based on configuration
function loadData() {
    const loadingIndicator = document.getElementById('loading-indicator');
    loadingIndicator.style.display = 'block'; // Show loading indicator

    if (typeof remoteDataUrl !== 'undefined' && remoteDataUrl !== '') {
        // Fetch data from remote URL
        const url = remoteDataUrl.endsWith('/') ? remoteDataUrl + 'data.json' : remoteDataUrl + '/data.json';
        fetchAndProcessData(url);
    } else {
        // Use local data
        loadedBenchmarkRuns = benchmarkRuns;
        initializeCharts();
        loadingIndicator.style.display = 'none'; // Hide loading indicator
    }
}

// Function to load archived data and merge with current data
// Archived data consists of older benchmark results that have been separated from
// the primary dataset but are still available for historical analysis.
function loadArchivedData() {
    const loadingIndicator = document.getElementById('loading-indicator');
    loadingIndicator.style.display = 'block';

    if (archivedDataLoaded) {
        updateCharts();
        loadingIndicator.style.display = 'none';
        return;
    }

    if (typeof remoteDataUrl !== 'undefined' && remoteDataUrl !== '') {
        // Fetch data from remote URL
        const url = remoteDataUrl.endsWith('/') ? remoteDataUrl + 'data_archive.json' : remoteDataUrl + '/data_archive.json';
        fetchAndProcessData(url, true);
    } else {
        // For local data use a static js file
        const script = document.createElement('script');
        script.src = 'data_archive.js';
        script.onload = () => {
            // Merge the archived runs with current runs
            loadedBenchmarkRuns = loadedBenchmarkRuns.concat(benchmarkRuns);
            archivedDataLoaded = true;
            initializeCharts();
            loadingIndicator.style.display = 'none';
        };

        script.onerror = () => {
            console.error('Failed to load data_archive.js');
            loadingIndicator.style.display = 'none';
        };

        document.head.appendChild(script);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    loadData();
});

// Process all benchmark runs to create a global options map for annotations
function createAnnotationsOptions() {
    const repoMap = new Map();

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            if (result.git_url && !repoMap.has(result.git_url)) {
                const suiteName = result.suite;
                const colorIndex = repoMap.size % annotationPalette.length;
                const backgroundColor = annotationPalette[colorIndex].replace('0.8', '0.9');
                const color = {
                    border: annotationPalette[colorIndex],
                    background: backgroundColor
                };

                repoMap.set(result.git_url, {
                    name: suiteName,
                    url: result.git_url,
                    color: color,
                });
            }
        });
    });

    return repoMap;
}

// Function to create chart data for flamegraph-only mode
function createFlameGraphOnlyData() {
    // Check if we have flamegraphData from data.js
    if (validateFlameGraphData()) {
        // Collect all unique benchmarks from all runs that have flamegraphs
        const allBenchmarks = new Set();
        const availableRuns = Object.keys(flamegraphData.runs);
        
        availableRuns.forEach(runName => {
            if (flamegraphData.runs[runName].benchmarks) {
                flamegraphData.runs[runName].benchmarks.forEach(benchmark => {
                    allBenchmarks.add(benchmark);
                });
            }
        });
        
        if (allBenchmarks.size > 0) {
            console.log(`Using flamegraphData from data.js for runs: ${availableRuns.join(', ')}`);
            console.log(`Available benchmarks with flamegraphs: ${Array.from(allBenchmarks).join(', ')}`);
            createSyntheticFlameGraphData(Array.from(allBenchmarks));
            return; // Success - we have flamegraph data
        }
    }
    
    // No flamegraph data available - benchmarks were run without --flamegraph option
    console.log('No flamegraph data found - benchmarks were likely run without --flamegraph option');
    
    // Disable the flamegraph checkbox since no flamegraphs are available
    const flameGraphToggle = document.getElementById('show-flamegraph');
    if (flameGraphToggle) {
        flameGraphToggle.disabled = true;
        flameGraphToggle.checked = false;
        
        // Add a visual indicator that flamegraphs are not available
        const label = document.querySelector('label[for="show-flamegraph"]');
        if (label) {
            label.style.color = '#999';
            label.title = 'No flamegraph data available - run benchmarks with --flamegraph option to enable';
        }
        
        console.log('Disabled flamegraph toggle - no flamegraph data available');
    }
    
    // Clear any flamegraph-only mode detection and proceed with normal benchmark display
    // This handles the case where we're in flamegraph-only mode but have no actual flamegraph data
}

function displayNoFlameGraphsMessage() {
    // Clear existing data arrays
    timeseriesData = [];
    barChartsData = [];
    layerComparisonsData = [];
    
    // Add a special suite for the message
    suiteNames.add('Information');
    
    // Create a special entry to show a helpful message
    const messageData = {
        label: 'No FlameGraphs Available',
        display_label: 'No FlameGraphs Available',
        suite: 'Information',
        unit: 'message',
        lower_is_better: false,
        range_min: null,
        range_max: null,
        runs: {}
    };
    
    timeseriesData.push(messageData);
    console.log('Added informational message about missing flamegraphs');
}

function displayNoDataMessage() {
    // Clear existing data arrays
    timeseriesData = [];
    barChartsData = [];
    layerComparisonsData = [];
    
    // Add a special suite for the message
    suiteNames.add('Information');
    
    // Create a special entry to show a helpful message
    const messageData = {
        label: 'No Data Available',
        display_label: 'No Benchmark Data Available',
        suite: 'Information',
        unit: 'message',
        lower_is_better: false,
        range_min: null,
        range_max: null,
        runs: {}
    };
    
    timeseriesData.push(messageData);
    console.log('Added informational message about missing benchmark data');
}

function createSyntheticFlameGraphData(flamegraphLabels) {
    // Clear existing data arrays since we're in flamegraph-only mode
    timeseriesData = [];
    barChartsData = [];
    layerComparisonsData = [];
    
    // Create synthetic benchmark results for each flamegraph
    flamegraphLabels.forEach(label => {
        // Try to determine suite from metadata, default to "Flamegraphs"
        const metadata = metadataForLabel(label, 'benchmark');
        let suite = 'Flamegraphs';
        
        // Try to match with existing metadata to get proper suite name
        if (metadata) {
            // Most flamegraphs are likely from Compute Benchmarks
            suite = 'Compute Benchmarks';
        } else {
            // For common benchmark patterns, assume Compute Benchmarks
            const computeBenchmarkPatterns = [
                'SubmitKernel', 'SubmitGraph', 'FinalizeGraph', 'SinKernelGraph',
                'AllocateBuffer', 'CopyBuffer', 'CopyImage', 'CreateBuffer',
                'CreateContext', 'CreateImage', 'CreateKernel', 'CreateProgram',
                'CreateQueue', 'ExecuteKernel', 'MapBuffer', 'MapImage',
                'ReadBuffer', 'ReadImage', 'WriteBuffer', 'WriteImage'
            ];
            
            if (computeBenchmarkPatterns.some(pattern => label.includes(pattern))) {
                suite = 'Compute Benchmarks';
            }
        }
        
        // Add to suite names
        suiteNames.add(suite);
        
        // Create a synthetic timeseries entry for this flamegraph
        const syntheticData = {
            label: label,
            display_label: metadata?.display_name || label,
            suite: suite,
            unit: 'flamegraph',
            lower_is_better: false,
            range_min: null,
            range_max: null,
            runs: {}
        };
        
        // Add this to timeseriesData so it shows up in the charts
        timeseriesData.push(syntheticData);
    });
    
    console.log(`Created synthetic data for ${flamegraphLabels.length} flamegraphs with suites:`, Array.from(suiteNames));
}

