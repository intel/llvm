// Copyright (C) 2024-2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core state
let activeRuns = new Set(defaultCompareNames);
let chartInstances = new Map();
let suiteNames = new Set();
let timeseriesData, barChartsData, allRunNames;

// DOM Elements
let runSelect, selectedRunsDiv, suiteFiltersContainer;

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
        plugins: {
            title: {
                display: true,
                text: data.label
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
                                `${data.label}:`,
                                `Value: ${point.y.toFixed(2)} ${data.unit}`,
                                `Stddev: ${point.stddev.toFixed(2)} ${data.unit}`,
                                `Git Hash: ${point.gitHash}`,
                            ];
                        } else {
                            return [`${context.dataset.label}:`,
                                `Value: ${context.parsed.y.toFixed(2)} ${data.unit}`,
                            ];
                        }
                    }
                }
            }
        },
        scales: {
            y: {
                title: {
                    display: true,
                    text: data.unit
                },
                grace: '20%',
            }
        }
    };

    if (type === 'time') {
        options.interaction = {
            mode: 'nearest',
            intersect: false
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
            type: 'time',
            ticks: {
                maxRotation: 45,
                minRotation: 45,
                autoSkip: true,
                maxTicksLimit: 10
            }
        };
    }

    const chartConfig = {
        type: type === 'time' ? 'line' : 'bar',
        data: type === 'time' ? {
            datasets: createTimeseriesDatasets(data)
        } : {
            labels: data.labels,
            datasets: data.datasets
        },
        options: options
    };

    const chart = new Chart(ctx, chartConfig);
    chartInstances.set(containerId, chart);
    return chart;
}

function createTimeseriesDatasets(data) {
    return Object.entries(data.runs).map(([name, points]) => ({
        label: name,
        data: points.map(p => ({
            x: new Date(p.date),
            y: p.value,
            gitHash: p.git_hash,
            gitRepo: p.github_repo,
            stddev: p.stddev
        })),
        borderWidth: 1,
        pointRadius: 3,
        pointStyle: 'circle',
        pointHoverRadius: 5
    }));
}

function updateCharts() {
    // Filter data by active runs
    const filteredTimeseriesData = timeseriesData.map(chart => ({
        ...chart,
        runs: Object.fromEntries(
            Object.entries(chart.runs).filter(([name]) => activeRuns.has(name))
        )
    }));

    const filteredBarChartsData = barChartsData.map(chart => ({
        ...chart,
        labels: chart.labels.filter(label => activeRuns.has(label)),
        datasets: chart.datasets.map(dataset => ({
            ...dataset,
            data: dataset.data.filter((_, i) => activeRuns.has(chart.labels[i]))
        }))
    }));

    // Draw charts with filtered data
    drawCharts(filteredTimeseriesData, filteredBarChartsData);
}

function drawCharts(filteredTimeseriesData, filteredBarChartsData) {
    // Clear existing charts
    document.querySelectorAll('.charts').forEach(container => container.innerHTML = '');
    chartInstances.forEach(chart => chart.destroy());
    chartInstances.clear();

    // Create timeseries charts
    filteredTimeseriesData.forEach((data, index) => {
        const containerId = `timeseries-${index}`;
        const container = createChartContainer(data, containerId, 'benchmark');
        document.querySelector('.timeseries .charts').appendChild(container);
        createChart(data, containerId, 'time');
    });

    // Create bar charts
    filteredBarChartsData.forEach((data, index) => {
        const containerId = `barchart-${index}`;
        const container = createChartContainer(data, containerId, 'group');
        document.querySelector('.bar-charts .charts').appendChild(container);
        createChart(data, containerId, 'bar');
    });

    // Apply current filters
    filterCharts();
}

function createChartContainer(data, canvasId, type) {
    const container = document.createElement('div');
    container.className = 'chart-container';
    container.setAttribute('data-label', data.label);
    container.setAttribute('data-suite', data.suite);

    // Check if this benchmark is marked as unstable
    const metadata = metadataForLabel(data.label, type);
    if (metadata && metadata.unstable) {
        container.setAttribute('data-unstable', 'true');

        // Add unstable warning
        const unstableWarning = document.createElement('div');
        unstableWarning.className = 'benchmark-unstable';
        unstableWarning.textContent = metadata.unstable;
        unstableWarning.style.display = isUnstableEnabled() ? 'block' : 'none';
        container.appendChild(unstableWarning);
    }

    // Add notes if present
    if (metadata && metadata.notes) {
        const noteElement = document.createElement('div');
        noteElement.className = 'benchmark-note';
        noteElement.textContent = metadata.notes;
        noteElement.style.display = isNotesEnabled() ? 'block' : 'none';
        container.appendChild(noteElement);
    }

    // Add description if present in metadata, but only for groups
    if (metadata && metadata.description && metadata.type === "group") {
        const descElement = document.createElement('div');
        descElement.className = 'benchmark-description';
        descElement.textContent = metadata.description;
        container.appendChild(descElement);
    }

    const canvas = document.createElement('canvas');
    canvas.id = canvasId;
    container.appendChild(canvas);

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
    latestRunsLookup = createLatestRunsLookup(benchmarkRuns);
    extraInfo.innerHTML = generateExtraInfo(latestRunsLookup, data, 'benchmark');
    details.appendChild(extraInfo);

    container.appendChild(details);

    return container;
}

function metadataForLabel(label, type) {
    for (const [key, metadata] of Object.entries(benchmarkMetadata)) {
        if (metadata.type === type && label.startsWith(key)) {
            return metadata;
        }
    }

    return null;
}

// Pre-compute a lookup for the latest run per label
function createLatestRunsLookup(benchmarkRuns) {
    const latestRunsMap = new Map();

    benchmarkRuns.forEach(run => {
        // Yes, we need to convert the date every time. I checked.
        const runDate = new Date(run.date);
        run.results.forEach(result => {
            const label = result.label;
            if (!latestRunsMap.has(label) || runDate > new Date(latestRunsMap.get(label).date)) {
                latestRunsMap.set(label, {
                    run,
                    result
                });
            }
        });
    });

    return latestRunsMap;
}

function generateExtraInfo(latestRunsLookup, data, type) {
    const labels = data.datasets ? data.datasets.map(dataset => dataset.label) : [data.label];

    return labels.map(label => {
        const metadata = metadataForLabel(label, type);
        const latestRun = latestRunsLookup.get(label);

        let html = '<div class="extra-info-entry">';

        if (metadata) {
            html += `<strong>${label}:</strong> ${formatCommand(latestRun.result)}<br>`;

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
            html += `<strong>${label}:</strong> No data available`;
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
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

function updateURL() {
    const url = new URL(window.location);
    const regex = document.getElementById('bench-filter').value;
    const activeSuites = getActiveSuites();
    const activeRunsList = Array.from(activeRuns);

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
    url.searchParams.set('notes', isNotesEnabled());
    url.searchParams.set('unstable', isUnstableEnabled());

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

        // Hide unstable benchmarks if showUnstable is false
        const shouldShow = regex.test(label) &&
            activeSuites.includes(suite) &&
            (isUnstableEnabled() || !isUnstable);

        container.style.display = shouldShow ? '' : 'none';
    });

    updateURL();
}

function getActiveSuites() {
    return Array.from(document.querySelectorAll('.suite-checkbox:checked'))
        .map(checkbox => checkbox.getAttribute('data-suite'));
}

// Data processing
function processTimeseriesData(benchmarkRuns) {
    const resultsByLabel = {};

    benchmarkRuns.forEach(run => {
        const runDate = run.date ? new Date(run.date) : null;
        run.results.forEach(result => {
            if (!resultsByLabel[result.label]) {
                resultsByLabel[result.label] = {
                    label: result.label,
                    suite: result.suite,
                    unit: result.unit,
                    lower_is_better: result.lower_is_better,
                    runs: {}
                };
            }

            if (!resultsByLabel[result.label].runs[run.name]) {
                resultsByLabel[result.label].runs[run.name] = [];
            }

            resultsByLabel[result.label].runs[run.name].push({
                date: runDate,
                value: result.value,
                stddev: result.stddev,
                git_hash: run.git_hash,
                github_repo: run.github_repo
            });
        });
    });

    return Object.values(resultsByLabel);
}

function processBarChartsData(benchmarkRuns) {
    const groupedResults = {};

    benchmarkRuns.reverse().forEach(run => {
        run.results.forEach(result => {
            if (!result.explicit_group) return;

            if (!groupedResults[result.explicit_group]) {
                // Look up group metadata
                const groupMetadata = metadataForLabel(result.explicit_group);

                groupedResults[result.explicit_group] = {
                    label: result.explicit_group,
                    suite: result.suite,
                    unit: result.unit,
                    lower_is_better: result.lower_is_better,
                    labels: [],
                    datasets: [],
                    // Add metadata if available
                    description: groupMetadata?.description || null,
                    notes: groupMetadata?.notes || null,
                    unstable: groupMetadata?.unstable || null
                };
            }

            const group = groupedResults[result.explicit_group];

            if (!group.labels.includes(run.name)) {
                group.labels.push(run.name);
            }

            let dataset = group.datasets.find(d => d.label === result.label);
            if (!dataset) {
                dataset = {
                    label: result.label,
                    data: new Array(group.labels.length).fill(null)
                };
                group.datasets.push(dataset);
            }

            const runIndex = group.labels.indexOf(run.name);
            dataset.data[runIndex] = result.value;
        });
    });

    return Object.values(groupedResults);
}

// Setup functions
function setupRunSelector() {
    runSelect = document.getElementById('run-select');
    selectedRunsDiv = document.getElementById('selected-runs');

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

    benchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            suiteNames.add(result.suite);
        });
    });

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

function setupToggles() {
    const notesToggle = document.getElementById('show-notes');
    const unstableToggle = document.getElementById('show-unstable');

    notesToggle.addEventListener('change', function() {
        // Update all note elements visibility
        document.querySelectorAll('.benchmark-note').forEach(note => {
            note.style.display = isNotesEnabled() ? 'block' : 'none';
        });
    });

    unstableToggle.addEventListener('change', function() {
        // Update all unstable warning elements visibility
        document.querySelectorAll('.benchmark-unstable').forEach(warning => {
            warning.style.display = isUnstableEnabled() ? 'block' : 'none';
        });
        filterCharts();
    });

    // Initialize from URL params if present
    const notesParam = getQueryParam('notes');
    const unstableParam = getQueryParam('unstable');

    if (notesParam !== null) {
        let showNotes = notesParam === 'true';
        notesToggle.checked = showNotes;
    }

    if (unstableParam !== null) {
        let showUnstable = unstableParam === 'true';
        unstableToggle.checked = showUnstable;
    }
}

function initializeCharts() {
    // Process raw data
    timeseriesData = processTimeseriesData(benchmarkRuns);
    barChartsData = processBarChartsData(benchmarkRuns);
    allRunNames = [...new Set(benchmarkRuns.map(run => run.name))];

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
    setupToggles();

    // Apply URL parameters
    const regexParam = getQueryParam('regex');
    const suitesParam = getQueryParam('suites');

    if (regexParam) {
        document.getElementById('bench-filter').value = regexParam;
    }

    if (suitesParam) {
        const suites = suitesParam.split(',');
        document.querySelectorAll('.suite-checkbox').forEach(checkbox => {
            checkbox.checked = suites.includes(checkbox.getAttribute('data-suite'));
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

// Load data based on configuration
function loadData() {
    const loadingIndicator = document.getElementById('loading-indicator');
    loadingIndicator.style.display = 'block'; // Show loading indicator

    if (typeof remoteDataUrl !== 'undefined' && remoteDataUrl !== '') {
        // Fetch data from remote URL
        fetch(remoteDataUrl)
            .then(response => response.json())
            .then(data => {
                benchmarkRuns = data.runs || data;
                benchmarkMetadata = data.metadata || benchmarkMetadata || {};
                initializeCharts();
            })
            .catch(error => {
                console.error('Error fetching remote data:', error);
                loadingIndicator.textContent = 'Fetching remote data failed.';
            })
            .finally(() => {
                loadingIndicator.style.display = 'none'; // Hide loading indicator
            });
    } else {
        // Use local data (benchmarkRuns and benchmarkMetadata should be defined in data.js)
        initializeCharts();
        loadingIndicator.style.display = 'none'; // Hide loading indicator
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    loadData();
});
