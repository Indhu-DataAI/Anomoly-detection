import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container,
  Typography,
  Paper,
  Box,
  Button,
  Alert,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  LinearProgress
} from '@mui/material';
import { Upload, Download, Analytics, Info } from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const API_BASE_URL = 'https://anomoly-detection-backend.onrender.com';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
}));

const FileUploadArea = styled(Box)(({ theme }) => ({
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(4),
  textAlign: 'center',
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
    borderColor: theme.palette.primary.dark,
  },
}));

const MetricCard = styled(Card)(({ theme }) => ({
  textAlign: 'center',
  padding: theme.spacing(2),
  height: '100%',
}));

function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('All');
  const [healthStatus, setHealthStatus] = useState('checking');

  useEffect(() => {
    checkHealth();
    fetchModelInfo();
    fetchFeatureImportance();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setHealthStatus(response.data.model_loaded ? 'healthy' : 'unhealthy');
    } catch (error) {
      setHealthStatus('unhealthy');
    }
  };

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model-info`);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Error fetching model info:', error);
    }
  };

  const fetchFeatureImportance = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/feature-importance`);
      setFeatureImportance(response.data.feature_importance);
    } catch (error) {
      console.error('Error fetching feature importance:', error);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setSelectedFile(file);
      setError('');
    } else {
      setError('Please select a valid CSV file');
      setSelectedFile(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setPredictions(response.data.data);
      } else {
        setError('Prediction failed');
      }
    } catch (error) {
      setError(error.response?.data?.detail || 'Error during prediction');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!predictions?.predictions) return;

    const csvContent = convertToCSV(predictions.predictions);
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'anomaly_detection_results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const convertToCSV = (data) => {
    if (!data || data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const csvHeaders = headers.join(',');
    const csvRows = data.map(row => 
      headers.map(header => {
        const value = row[header];
        return typeof value === 'string' ? `"${value}"` : value;
      }).join(',')
    );
    
    return [csvHeaders, ...csvRows].join('\n');
  };

  const getFilteredPredictions = () => {
    if (!predictions?.predictions || filter === 'All') {
      return predictions?.predictions || [];
    }
    return predictions.predictions.filter(item => item.prediction === filter);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'unhealthy': return 'error';
      default: return 'warning';
    }
  };

  const getPredictionColor = (prediction) => {
    return prediction && prediction.toLowerCase().includes('normal') ? 'success' : 'error';
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom>
          🔍 Smart System Anomaly Detection
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Upload your CSV file to detect anomalies in smart system data
        </Typography>
        <Chip 
          label={`Status: ${healthStatus}`} 
          color={getStatusColor(healthStatus)}
          sx={{ mt: 2 }}
        />
      </Box>

      <Grid container spacing={3}>
        {/* Left Column - Model Info */}
        <Grid item xs={12} md={4}>
          

          {/* Feature Importance */}
          {featureImportance.length > 0 && (
            <StyledPaper>
              <Typography variant="h6" gutterBottom>
                <Analytics sx={{ mr: 1, verticalAlign: 'middle' }} />
                Top Feature Importance
              </Typography>
              <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                {featureImportance.map((item, index) => (
                  <Box key={index} sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2">{item.feature}</Typography>
                      <Typography variant="body2">{item.importance.toFixed(4)}</Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={(item.importance / Math.max(...featureImportance.map(f => f.importance))) * 100}
                      sx={{ height: 4, borderRadius: 2 }}
                    />
                  </Box>
                ))}
              </Box>
            </StyledPaper>
          )}
        </Grid>

        {/* Right Column - File Upload and Results */}
        <Grid item xs={12} md={8}>
          {/* File Upload */}
          <StyledPaper>
            <Typography variant="h6" gutterBottom>
              📂 Upload CSV File
            </Typography>
            
            <input
              accept=".csv"
              style={{ display: 'none' }}
              id="file-upload"
              type="file"
              onChange={handleFileSelect}
            />
            <label htmlFor="file-upload">
              <FileUploadArea>
                <Upload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {selectedFile ? selectedFile.name : 'Click to upload CSV file'}
                </Typography>
                <Typography color="text.secondary">
                  {selectedFile ? `Size: ${(selectedFile.size / 1024).toFixed(1)} KB` : 'Select a CSV file to analyze'}
                </Typography>
              </FileUploadArea>
            </label>

            <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                onClick={handlePredict}
                disabled={!selectedFile || loading}
                startIcon={loading ? <CircularProgress size={20} /> : <Analytics />}
              >
                {loading ? 'Analyzing...' : 'Detect Anomalies'}
              </Button>
              
              {predictions && (
                <Button
                  variant="outlined"
                  onClick={handleDownload}
                  startIcon={<Download />}
                >
                  Download Results
                </Button>
              )}
            </Box>

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </StyledPaper>

          {/* Results */}
          {predictions && (
            <>
              {/* Statistics Cards */}
              <StyledPaper>
                <Typography variant="h6" gutterBottom>
                  🎯 Prediction Results
                </Typography>
                <Grid container spacing={2}>
                  {predictions.prediction_stats.map((stat, index) => (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                      <MetricCard>
                        <CardContent>
                          <Typography variant="h4" color={getPredictionColor(stat.class)}>
                            {stat.count}
                          </Typography>
                          <Typography variant="h6" gutterBottom>
                            {stat.class}
                          </Typography>
                          <Chip 
                            label={`${stat.percentage}%`} 
                            color={getPredictionColor(stat.class)}
                            variant="outlined"
                          />
                        </CardContent>
                      </MetricCard>
                    </Grid>
                  ))}
                </Grid>

                {/* Confidence Stats */}
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>Confidence Statistics:</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={3}>
                      <Typography variant="body2">Mean: {predictions.confidence_distribution.mean.toFixed(3)}</Typography>
                    </Grid>
                    <Grid item xs={3}>
                      <Typography variant="body2">Std: {predictions.confidence_distribution.std.toFixed(3)}</Typography>
                    </Grid>
                    <Grid item xs={3}>
                      <Typography variant="body2">Min: {predictions.confidence_distribution.min.toFixed(3)}</Typography>
                    </Grid>
                    <Grid item xs={3}>
                      <Typography variant="body2">Max: {predictions.confidence_distribution.max.toFixed(3)}</Typography>
                    </Grid>
                  </Grid>
                </Box>
              </StyledPaper>

              {/* Detailed Results Table */}
              <StyledPaper>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">📊 Detailed Results</Typography>
                  <FormControl sx={{ minWidth: 150 }}>
                    <InputLabel>Filter by prediction</InputLabel>
                    <Select
                      value={filter}
                      label="Filter by prediction"
                      onChange={(e) => setFilter(e.target.value)}
                    >
                      <MenuItem value="All">All</MenuItem>
                      <MenuItem value="Anomaly_DoS">Anomaly_DoS</MenuItem>
                      <MenuItem value="Anomaly_Spoofing">Anomaly_Probe</MenuItem>
                      <MenuItem value="Anomaly_Injection">Anomaly_R2L</MenuItem>
                      {modelInfo?.classes.map((cls) => (
                        <MenuItem key={cls} value={cls}>{cls}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>

                <Typography variant="body2" sx={{ mb: 2 }}>
                  Showing {getFilteredPredictions().length} / {predictions.total_records} records
                </Typography>

                <TableContainer sx={{ maxHeight: 600 }}>
                  <Table stickyHeader>
                    <TableHead>
                      <TableRow>
                        {predictions.predictions.length > 0 && 
                          Object.keys(predictions.predictions[0]).map((key) => (
                            <TableCell key={key} sx={{ fontWeight: 'bold' }}>
                              {key.charAt(0).toUpperCase() + key.slice(1)}
                            </TableCell>
                          ))
                        }
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {getFilteredPredictions().slice(0, 100).map((row, index) => (
                        <TableRow key={index} hover>
                          {Object.entries(row).map(([key, value]) => (
                            <TableCell key={key}>
                              {key === 'prediction' ? (
                                <Chip 
                                  label={value} 
                                  color={getPredictionColor(value)}
                                  size="small"
                                />
                              ) : key === 'confidence' ? (
                                <Box>
                                  {typeof value === 'number' ? value.toFixed(3) : value}
                                  <LinearProgress
                                    variant="determinate"
                                    value={typeof value === 'number' ? value * 100 : 0}
                                    sx={{ mt: 0.5, height: 4 }}
                                  />
                                </Box>
                              ) : (
                                typeof value === 'number' ? value.toFixed(3) : value
                              )}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                {getFilteredPredictions().length > 100 && (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    Showing first 100 records. Download the full results using the Download button.
                  </Alert>
                )}
              </StyledPaper>
            </>
          )}
        </Grid>
      </Grid>
    </Container>
  );
}

export default App;