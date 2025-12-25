use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::math::primitives::RegularPolygon;

// Components 

#[derive(Component)]
struct Boid;

#[derive(Component)]
struct BoidCamera;

#[derive(Component)]
struct Velocity(Vec2);

#[derive(Component)]
struct Acceleration(Vec2);

#[derive(Clone, Copy, Debug)]
struct BoidState {
    entity: Entity,
    pos: Vec2,
    vel: Vec2,
}

type Snapshot = Vec<BoidState>;

// Constants

const LEFT: f32 = -400.0;
const RIGHT: f32 = 400.0;
const TOP: f32 = 300.0;
const BOTTOM: f32 = -300.0;

const TURN_FACTOR: f32 = 50.0;
const MAX_SPEED: f32 = 200.0;
const MIN_SPEED: f32 = 50.0;

const NUM_BOIDS: usize = 80;

const PERCEPTION_RADIUS: f32 = 80.0;
const SEPARATION_RADIUS: f32 = 30.0;
const MAX_FORCE: f32 = 120.0;

const ALIGN_WEIGHT: f32 = 1.0;
const COHESION_WEIGHT: f32 = 0.8;
const SEPARATION_WEIGHT: f32 = 1.4;

const EPS: f32 = 1e-5;

//
fn clamp_len(v: Vec2, max: f32) -> Vec2 {
    let len = v.length();
    if len > max && len > EPS {
        v / len * max
    } else {
        v
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((
        Camera2dBundle::default(),
        BoidCamera,
    ));
    let triangle = Mesh::from(RegularPolygon::new(10.0, 3));
    let msh = meshes.add(triangle);
    let mat = materials.add(Color::WHITE);
    for i in 0..NUM_BOIDS {
        let x = ((i as f32) * 37.0).sin() * 300.0;
        let y = ((i as f32) * 53.0).cos() * 200.0;
        let vx = ((i as f32) * 11.0).cos() * 120.0;
        let vy = ((i as f32) * 17.0).sin() * 120.0;

        commands.spawn((
            Boid,
            Velocity(Vec2::new(vx, vy)),
            Acceleration(Vec2::ZERO),
            MaterialMesh2dBundle {
                mesh: msh.clone().into(),
                material: mat.clone(),
                transform: Transform::from_xyz(x, y, 0.0),
                ..Default::default()
            },
            ..Default::default()
        ));
    }
}

fn snapshot_boids(query: &Query<(Entity, &Transform, &Velocity), With<Boid>>) -> Snapshot {
    query
        .iter()
        .map(|(entity, t, v)| BoidState {
            entity,
            pos: t.translation.truncate(),
            vel: v.0,
        })
        .collect()
}

fn boundary_system(
    mut query: Query<(&mut Transform, &mut Velocity), 
    With<Boid>>
) {
    for (mut transform, mut velocity) in query.iter_mut() {
        let pos = transform.translation;

        if pos.x < LEFT {
            velocity.0.x += TURN_FACTOR;
        } 
        else if pos.x > RIGHT {
            velocity.0.x -= TURN_FACTOR;
        }
        if pos.y < BOTTOM {
            velocity.0.y += TURN_FACTOR;
        } 
        else if pos.y > TOP {
            velocity.0.y -= TURN_FACTOR;
        }
    }
}

fn integrate_system(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut Velocity, &mut Acceleration), 
    With<Boid>>,
) {
    for (mut transform, mut velocity, mut acceleration) in &mut query {
        velocity.0 += acceleration.0 * time.delta_seconds();
        let speed = velocity.0.length();
        if speed > MAX_SPEED {
            velocity.0 = (velocity.0 / speed) * MAX_SPEED;
        }
        if speed < MIN_SPEED {
            velocity.0 = (velocity.0 / speed) * MIN_SPEED;
        }
        transform.translation += velocity.0.extend(0.0) * time.delta_seconds();
        acceleration.0 = Vec2::ZERO;
    }
}

fn separation_system(
    read: Query<(Entity, &Transform, &Velocity), With<Boid>>,
    mut write: Query<(Entity, &Transform, &Velocity, &mut Acceleration), With<Boid>>,
) {
    let snapshot = snapshot_boids(&read);
    let sr2 = SEPARATION_RADIUS * SEPARATION_RADIUS;

    for (entity, transform, velocity, mut acceleration) in write.iter_mut() {
        let pos = transform.translation.truncate();

        let mut away_sum = Vec2::ZERO;
        let mut count = 0.0;

        for other in snapshot.iter() {
            if other.entity == entity {
                continue;
            }
            let offset = pos - other.pos;
            let dist2 = offset.length_squared();

            if dist2 < sr2 && dist2 > EPS {
                away_sum += offset / dist2.sqrt(); 
                count += 1.0;
            }
        }

        if count > 0.0 {
            let away = away_sum / count;
            let desired = away.normalize_or_zero() * MAX_SPEED;
            let steer = clamp_len(desired - velocity.0, MAX_FORCE);
            acceleration.0 += steer * SEPARATION_WEIGHT;
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (
            boundary_system,
            integrate_system,
        ))
        .run();
}