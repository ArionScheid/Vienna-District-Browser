import {ComponentFixture, TestBed} from '@angular/core/testing';
import {HttpClientTestingModule} from '@angular/common/http/testing';
import {FormsModule} from '@angular/forms';
import {MapComponent} from './map.component';

describe('MapComponent', () => {
  let component: MapComponent;
  let fixture: ComponentFixture<MapComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HttpClientTestingModule, FormsModule],
      declarations: [MapComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(MapComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('ranks districts with scores within [0, 1]', () => {
    const f = (a: number | null, b: number, c: number) => ({components: {affordability: a, accessibility: b, transit: c}, name: 'X'} as any);
    component.features = {'1': f(null, 1, 1), '2': f(1, 0.2, 0.3), '3': f(0, 0, 0)};
    component.weights = {affordability: 10, accessibility: 10, transit: 10};
    component.computeScores();
    expect(component.ranking.length).toBe(3);
    expect(component.ranking.every(r => r.score >= 0 && r.score <= 1)).toBeTrue();
    expect(component.ranking[2].district).toBe(3);
  });
});
